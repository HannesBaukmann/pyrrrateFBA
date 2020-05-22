import numpy as np
from collections import OrderedDict

try:
    import libsbml as sbml
except ImportError as err:
    raise ImportError("SBML support requires the libsbml module, but importing this module failed with message: " + err)


class SBMLError(Exception):
    """
    empty error class to state that something with the import of the SBML file gone wrong
    """
    pass


class RAMError(Exception):
    """
    empty error class to state that something with the import of the RAM annotations gone wrong
    """
    pass


def readSBML(filename):
    """
    Convert SBML file to an deFBA model object.
    Required argument:
    - filename              string. Full name of the .xml file, which you want to import.
    """
    reader = sbml.SBMLReader()
    document = reader.readSBML(filename)
    if document.isSetModel():  # Returns True if the Model object has been set.
        # Initialize RAMParser object
        parsed = RAMParser(document)
        # return the model object
        import pyrrrateModel
        model = pyrrrateModel.Model(parsed.stoich, parsed.name, metabolites=parsed.metabolites_dict,
                                    macromolecules=parsed.macromolecules_dict, reactions=parsed.reactions_dict,
                                    HC=parsed.HC_matrix, HE=parsed.HE_matrix, HB=parsed.HB_matrix, HM=parsed.HM_matrix)
        return model
    else:
        raise SBMLError(
            'The SBML file contains no model. Maybe the filename is wrong or the file does not follow SBML standards. Please run the SBML validator at http://sbml.org/Facilities/Validator/index.jsp to find the problem.')


class RAMParser:
    """
    read all necessary information from a SBML file supporting the Resource Allocation Modelling (RAM) annotation standard and convert them
    to the matrix representation of a deFBA model. Minimimal informationen content is the stoichiometric matrix and the molecular weights of
    objective species (macromolecules)
    """

    def __init__(self, document):
        """
        Required arguments:
        - document      libsbml.reader object containing the SBML data
        """
        self.name = None
        self.metabolites_dict = OrderedDict()
        self.macromolecules_dict = OrderedDict()
        self.reactions_dict = OrderedDict()

        # MATRICES
        self.HC_matrix = None  # Enzyme Capacity Constraint matrix
        self.HE_matrix = None  # Filter matrix for ECC matrix
        self.HM_matrix = None  # Maintenance matrix
        self.HB_matrix = None  # Biomass composition constraints

        # MODEL
        model = document.getModel()  # Returns the Model contained in this SBMLDocument, or None if no such model exists.
        if not model:
            raise SBMLError(
                'The SBML file contains no model. Maybe the filename is wrong or the file does not follow SBML standards. Please run the SBML validator at http://sbml.org/Facilities/Validator/index.jsp to find the problem.')

        self.name = model.getId()

        # qual_model = model.getPlugin('qual')

        # SPECIES
        for s in (model.species):
            s_id = s.getId()
            if s_id in self.metabolites_dict.keys() or s_id in self.macromolecules_dict.keys():
                # funktioniert das so?
                raise SBMLError('The species id ' + s_id + ' is not unique!')

            # get RAM species attributes
            annotation = s.getAnnotation()
            if annotation:
                # Because annotations of other types can be present we need to look at each annotation individually to find the RAM element
                ram_element = ''
                for child_number in range(annotation.getNumChildren()):
                    child = annotation.getChild(child_number)
                    if child.getName() == 'RAM':
                        url = child.getURI()  # XML namespace URI of the attribute.
                        ram_element = child.getChild(0)
                        break
                if ram_element:  # False if string is empty
                    s_type = ram_element.getAttrValue('speciesType', url)
                    if s_type == 'metabolite' or s_type == 'extracellular':
                        self.metabolites_dict[s_id] = {}
                        self.metabolites_dict[s_id]['speciesType'] = s_type
                    elif s_type == 'enzyme' or s_type == 'quota' or s_type == 'storage':
                        self.macromolecules_dict[s_id] = {}
                        self.macromolecules_dict[s_id]['speciesType'] = s_type
                    else:
                        raise RAMError('unknown species type ' + s_type + ' found in the RAM annotation ' + s_id)
                    # or check consistency later when the species dictionary has been completed?

                    # try to import the molecular weight (can be a string pointing to a paramter, int, or double)
                    try:
                        weight = float(ram_element.getAttrValue('molecularWeight', url))
                    except ValueError:
                        weight_str = ram_element.getAttrValue('molecularWeight', url)
                        if weight_str:
                            try:
                                weight = float(model.getParameter(weight_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + weight_str + ' has no value.')
                        else:
                            if s_type == 'extracellular' or s_type == 'metabolite':
                                weight = 0.0  # default for metabolites
                            else:
                                raise RAMError(
                                    'The molecular weight of species ' + s_id + ' is not set althought it is supposed to be a biomass species. Please correct the error in the SBML file')

                    # try to import the objective weight (can be a string pointing to a paramter, int, or double)
                    try:
                        oweight = float(ram_element.getAttrValue('objectiveWeight', url))
                    except ValueError:
                        oweight_str = ram_element.getAttrValue('objectiveWeight', url)
                        if oweight_str:
                            try:
                                oweight = float(model.getParameter(oweight_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + oweight_str + ' has no value.')
                        else:
                            if s_type == 'extracellular' or s_type == 'metabolite':
                                oweight = 0.0  # default for metabolites
                            else:
                                raise RAMError(
                                    'The objective weight of species ' + s_id + ' is not set althought it is supposed to be a biomass species. Please correct the error in the SBML file')
                    if s_type == 'metabolite' or s_type == 'extracellular':
                        self.metabolites_dict[s_id]['molecularWeight'] = weight
                        self.metabolites_dict[s_id]['objectiveWeight'] = oweight
                    elif s_type == 'enzyme' or s_type == 'quota' or s_type == 'storage':
                        self.macromolecules_dict[s_id]['molecularWeight'] = weight
                        self.macromolecules_dict[s_id]['objectiveWeight'] = oweight

                    # Try to import the biomass percentage for quota macromolecules
                    if s_type == "quota":
                        try:
                            biomass = float(ram_element.getAttrValue('biomassPercentage', url))
                        except ValueError:
                            biomass_string = ram_element.getAttrValue('biomassPercentage', url)
                            if biomass_string:
                                try:
                                    biomass = float(model.getParameter(biomp_string).getValue())
                                except AttributeError:
                                    print('The parameter ' + biomass_string + ' has no value.')
                        if biomass < 0 or biomass > 1:
                            raise RAMError('The parameter ' + biomp_string + ' does not have a value between 0 and 1.')
                        self.macromolecules_dict[s_id]['biomassPercentage'] = biomass
                        # Hinweis, dass man nicht kontrolliert, ob im Modell eine biomassP für eine nicht-quota species steht?

                else:  # no RAM elements
                    raise SBMLError(
                        'Species ' + s_id + ' does not have a RAM annotation. Stopping import.')
            else:  # no annotation
                raise SBMLError(
                    'Species ' + s_id + ' does not have a RAM annotation. Stopping import.')

            # get species attributes
            if s_type == 'metabolite' or s_type == 'extracellular':
                self.metabolites_dict[s_id]['name'] = s.getName()
                self.metabolites_dict[s_id]['compartment'] = s.getCompartment()
                self.metabolites_dict[s_id]['initialAmount'] = s.getInitialAmount()
                self.metabolites_dict[s_id]['constant'] = s.getConstant()
                self.metabolites_dict[s_id]['boundaryCondition'] = s.getBoundaryCondition()
                self.metabolites_dict[s_id]['hasOnlySubstanceUnits'] = s.getHasOnlySubstanceUnits()
            elif s_type == 'enzyme' or s_type == 'quota' or s_type == 'storage':
                self.macromolecules_dict[s_id]['name'] = s.getName()
                self.macromolecules_dict[s_id]['compartment'] = s.getCompartment()
                self.macromolecules_dict[s_id]['initialAmount'] = s.getInitialAmount()
                self.macromolecules_dict[s_id]['constant'] = s.getConstant()
                self.macromolecules_dict[s_id]['boundaryCondition'] = s.getBoundaryCondition()
                self.macromolecules_dict[s_id]['hasOnlySubstanceUnits'] = s.getHasOnlySubstanceUnits()

        # REACTIONS
        self.stoich = np.zeros(
            (len(self.metabolites_dict), model.getNumReactions()))  # stoich is the stoichiometric matrix
        # Loop over all reactions. gather stoichiometry, reversibility, kcats and gene associations
        for r in model.reactions:
            r_id = r.getId()
            if r_id in self.reactions_dict:
                raise SBMLError('The reaction id ' + r_id + ' is not unique!')
            self.reactions_dict[r_id] = {}
            # get reaction attributes
            self.reactions_dict[r_id]['reversible'] = r.getReversible()
            #            self.reactions_dict[r_id]['fast'] = False

            # get gene association
            fbc_model = model.getPlugin('fbc')
            reaction_fbc = r.getPlugin('fbc')
            # (geht das irgendwie eleganter?)
            if reaction_fbc.getGeneProductAssociation():
                try:
                    gene_product_id = reaction_fbc.getGeneProductAssociation().all_elements[0].getGeneProduct()
                    gene_product = fbc_model.getGeneProduct(gene_product_id)  # object
                    enzyme = gene_product.getAssociatedSpecies()
                    if enzyme == '':
                        if gene_product_id in self.macromolecules_dict.keys():
                            self.reactions_dict[r_id]['geneProduct'] = gene_product_id
                        else:
                            raise RAMError('The reaction ' + r_id + ' has an empty fbc:geneProductRef()')
                    else:
                        if enzyme in self.macromolecules_dict.keys():
                            self.reactions_dict[r_id]['geneProduct'] = enzyme
                        else:
                            raise RAMError(
                                'fbc:geneAssociation for geneProduct ' + gene_product_id + ' is pointing to an unknown species')
                except ValueError:
                    print('No gene product association given for reaction ' + r_id)
            else:
                self.reactions_dict[r_id]['geneProduct'] = None

            # get flux balance constraints
            if reaction_fbc.getLowerFluxBound():
                self.reactions_dict[r_id]['lowerFluxBound'] = reaction_fbc.getLowerFluxBound()
            if reaction_fbc.getUpperFluxBound():
                self.reactions_dict[r_id]['upperFluxBound'] = reaction_fbc.getUpperFluxBound()

            # get RAM reactions attributes
            annotation = r.getAnnotation()
            if annotation:
                # Because annotations of other types can be present we need to look at each annotation individually to find the RAM element
                ram_element = ''
                for child_number in range(annotation.getNumChildren()):
                    child = annotation.getChild(child_number)
                    if child.getName() == 'RAM':
                        url = child.getURI()  # XML namespace URI of the attribute.
                        ram_element = child.getChild(0)
                        break
                if ram_element:  # False if string is empty
                    # try to import absolute value for scaling of maintenance reactions
                    main = 0.0  # default
                    try:
                        main = float(ram_element.getAttrValue('maintenanceScaling', url))
                    except ValueError:
                        main_str = ram_element.getAttrValue('maintenanceScaling', url)
                        if main_str:
                            try:
                                main = float(model.getParameter(main_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + main_str + ' has no value.')
                    self.reactions_dict[r_id]['maintenanceScaling'] = main

                    # Import forward kcat values
                    try:
                        k_fwd = float(ram_element.getAttrValue('kcatForward', url))
                    except ValueError:
                        k_fwd_str = ram_element.getAttrValue('kcatForward', url)
                        if k_fwd_str:
                            try:
                                k_fwd = float(model.getParameter(k_fwd_str).getValue())
                            except AttributeError:
                                raise RAMError('The parameter ' + k_fwd_str + ' has no value.')
                    self.reactions_dict[r_id]['kcatForward'] = k_fwd

                    # Import backward kcat values
                    if self.reactions_dict[r_id]['reversible'] == True:
                        try:
                            k_bwd = float(ram_element.getAttrValue('kcatBackward', url))
                        except ValueError:
                            k_bwd_str = ram_element.getAttrValue('kcatBackward', url)
                            if k_bwd_str:
                                try:
                                    k_bwd = float(model.getParameter(k_bwd_str).getValue())
                                except AttributeError:
                                    raise RAMError('The parameter ' + k_bwd_str + ' has no value.')
                        self.reactions_dict[r_id]['kcatBackward'] = k_bwd
                    else:
                        self.reactions_dict[r_id]['kcatBackward'] = 0.0

            if self.reactions_dict[r_id]['kcatForward'] == 0 and self.reactions_dict[r_id]['kcatBackward'] != 0:
                raise RAMError('The reaction ' + r_id + ' has no forward kcat value but a non-zero backward kcat. ')

            # fill stoichiometric matrix
            # exclude quota reactions?
            j = list(model.reactions).index(r)
            for educt in r.getListOfReactants():
                # only for metabolites
                if educt.getSpecies() in self.metabolites_dict.keys():
                    i = list(self.metabolites_dict).index(educt.getSpecies())
                    self.stoich[i, j] -= educt.getStoichiometry()
            for product in r.getListOfProducts():
                if product.getSpecies() in self.metabolites_dict.keys():
                    i = list(self.metabolites_dict).index(product.getSpecies())
                    self.stoich[i, j] += product.getStoichiometry()

        # delete boundary species from stoichiometric matrix (they are not modelled dynamically)
        boundary_id = []
        for met in self.metabolites_dict:
            if self.metabolites_dict[met]['compartment'] == 'extracellular':
                #                if self.metabolites_dict[met]['constant'] or self.metabolites_dict[met]['boundaryCondition']:
                boundary_id.append(list(self.metabolites_dict).index(met))
        self.stoich = np.delete(self.stoich, boundary_id, axis=0)