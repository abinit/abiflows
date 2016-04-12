from __future__ import print_function, division, unicode_literals

import logging
from fireworks.core.firework import FireTaskBase
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.matproj.rest import MPRester

import json


class ChemEnvStructureEnvironmentsTask(FireTaskBase):
    _fw_name = "ChemEnvStructureEnvironmentsTask"

    def run_task(self, fw_spec):
        logging.basicConfig(filename='chemenv_structure_environments.log',
                            format='%(levelname)s:%(module)s:%(funcName)s:%(message)s',
                            level=logging.DEBUG)
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                             structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)
        if 'chemenv_parameters' in fw_spec:
            for param, value in fw_spec['chemenv_parameters'].items():
                lgf.setup_parameter(param, value)
        identifier = fw_spec['identifier']
        if 'structure' in fw_spec:
            structure = fw_spec['structure']
        else:
            if identifier['source'] == 'MaterialsProject' and 'material_id' in identifier:
                if not 'mapi_key' in fw_spec:
                    raise ValueError('The mapi_key should be provided to get the structure from the Materials Project')
                a = MPRester(fw_spec['mapi_key'])
                structure = a.get_structure_by_material_id(identifier['material_id'])
            else:
                raise ValueError('Either structure or identifier with source = MaterialsProject and material_id '
                                 'should be provided')
        lgf.setup_structure(structure)
        se = lgf.compute_structure_environments_detailed_voronoi()
        if 'json_file' in fw_spec:
            json_file = fw_spec['json_file']
        else:
            json_file = 'structure_environments.json'
        f = open(json_file, 'w')
        json.dump(se.as_dict(), f)
        f.close()
        if 'mongo_database' in fw_spec:
            database = fw_spec['mongo_database']
            entry = {'identifier': identifier,
                     'elements': [elmt.symbol for elmt in structure.composition.elements],
                     'nelements': len(structure.composition.elements),
                     'pretty_formula': structure.composition.reduced_formula,
                     'nsites': len(structure)
                     }
            gridfs_msonables = {'structure': structure,
                                'structure_environments': se}
            criteria = {'identifier': identifier}
            if database.collection.find(criteria).count() == 1:
                database.update_entry(query=criteria, entry_update=entry,
                                      gridfs_msonables=gridfs_msonables)
            else:
                database.insert_entry(entry=entry, gridfs_msonables=gridfs_msonables)

class ChemEnvLightStructureEnvironmentsTask(FireTaskBase):
    _fw_name = "ChemEnvLightStructureEnvironmentsTask"

    def run_task(self, fw_spec):
        logging.basicConfig(filename='chemenv_light_structure_environments.log',
                            format='%(levelname)s:%(module)s:%(funcName)s:%(message)s',
                            level=logging.DEBUG)
        lgf = LocalGeometryFinder()
        lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                             structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)
        if 'chemenv_parameters' in fw_spec:
            for param, value in fw_spec['chemenv_parameters'].items():
                lgf.setup_parameter(param, value)
        structure = fw_spec['structure']
        identifier = fw_spec['identifier']
        lgf.setup_structure(structure)
        se = lgf.compute_structure_environments_detailed_voronoi()
        if 'json_file' in fw_spec:
            json_file = fw_spec['json_file']
        else:
            json_file = 'structure_environments.json'
        f = open(json_file, 'w')
        json.dump(se.as_dict(), f)
        f.close()
        if 'mongo_database' in fw_spec:
            database = fw_spec['mongo_database']
            entry = {'identifier': identifier,
                     'elements': [elmt.symbol for elmt in structure.composition.elements],
                     'nelements': len(structure.composition.elements),
                     'pretty_formula': structure.composition.reduced_formula,
                     'nsites': len(structure)
                     }
            gridfs_msonables = {'structure': structure.as_dict(),
                                'structure_environments': se.as_dict()}
            if 'criteria' in fw_spec:
                database.update_entry(query=fw_spec['criteria'], entry_update=entry,
                                      gridfs_msonables=gridfs_msonables)
            else:
                database.insert_entry(entry=entry, gridfs_msonables=gridfs_msonables)