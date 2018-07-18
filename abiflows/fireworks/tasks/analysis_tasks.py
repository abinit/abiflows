from __future__ import print_function, division, unicode_literals, absolute_import

import logging
import json

from fireworks.core.firework import FireTaskBase
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import StructureEnvironments
from pymatgen.analysis.bond_valence import BVAnalyzer
try:
    from pymatgen.ext.matproj import MPRester, MPRestError
except ImportError:
    from pymatgen.matproj.rest import MPRester, MPRestError


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

        info = {}
        # Compute the structure environments
        lgf.setup_structure(structure)
        if 'valences' in fw_spec:
            valences = fw_spec['valences']
        else:
            try:
                bva = BVAnalyzer()
                valences = bva.get_valences(structure=structure)
                info['valences'] = {'origin': 'BVAnalyzer'}
            except:
                valences = 'undefined'
                info['valences'] = {'origin': 'None'}
        excluded_atoms = None
        if 'excluded_atoms' in fw_spec:
            excluded_atoms = fw_spec['excluded_atoms']

        se = lgf.compute_structure_environments(only_cations=False, valences=valences, excluded_atoms=excluded_atoms)

        # Write to json file
        if 'json_file' in fw_spec:
            json_file = fw_spec['json_file']
        else:
            json_file = 'structure_environments.json'
        f = open(json_file, 'w')
        json.dump(se.as_dict(), f)
        f.close()

        # Save to database
        if 'mongo_database' in fw_spec:
            database = fw_spec['mongo_database']
            entry = {'identifier': identifier,
                     'elements': [elmt.symbol for elmt in structure.composition.elements],
                     'nelements': len(structure.composition.elements),
                     'pretty_formula': structure.composition.reduced_formula,
                     'nsites': len(structure)
                     }

            saving_option = fw_spec['saving_option']
            if saving_option == 'gridfs':
                gridfs_msonables = {'structure': structure,
                                    'structure_environments': se}
            elif saving_option == 'storefile':
                gridfs_msonables = None
                if 'se_prefix' in fw_spec:
                    se_prefix = fw_spec['se_prefix']
                    if not se_prefix.isalpha():
                        raise ValueError('Prefix for structure_environments file is "{}" '
                                         'while it should be alphabetic'.format(se_prefix))
                else:
                    se_prefix = ''
                if se_prefix:
                    se_rfilename = '{}_{}.json'.format(se_prefix, fw_spec['storefile_basename'])
                else:
                    se_rfilename = '{}.json'.format(fw_spec['storefile_basename'])
                se_rfilepath = '{}/{}'.format(fw_spec['storefile_dirpath'], se_rfilename)
                storage_server = fw_spec['storage_server']
                storage_server.put(localpath=json_file, remotepath=se_rfilepath, overwrite=True, makedirs=False)
                entry['structure_environments_file'] = se_rfilepath
            else:
                raise ValueError('Saving option is "{}" while it should be '
                                 '"gridfs" or "storefile"'.format(saving_option))
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

        identifier = fw_spec['identifier']
        criteria = {'identifier': identifier}
        # Where to get the full structure environments object
        se_database = fw_spec['structure_environments_database']
        entry = se_database.collection.find_one(criteria)
        if fw_spec['structure_environments_setup'] == 'from_gridfs':
            gfs_fileobject = se_database.gridfs.get(entry['structure_environments'])
            dd = json.load(gfs_fileobject)
            se = StructureEnvironments.from_dict(dd)
        elif fw_spec['structure_environments_setup'] == 'from_storedfile':
            se_storage_server = fw_spec['se_storage_server']
            se_filepath = entry['structure_environments_file']
            se_storage_server.get(se_filepath, 'se.json')
            f = open('se.json', 'r')
            dd = json.load(f)
            f.close()
            se = StructureEnvironments.from_dict(dd)
        else:
            raise RuntimeError('Wrong structure_environments_setup : '
                               '"{}" is not allowed'.format(fw_spec['structure_environments_setup']))

        # Compute the light structure environments
        chemenv_strategy = fw_spec['chemenv_strategy']
        if 'valences' in fw_spec:
            valences = fw_spec['valences']
            valences_origin = fw_spec['valences_origin']
        else:
            valences = 'undefined'
            valences_origin = 'None'
        lse = LightStructureEnvironments.from_structure_environments(strategy=chemenv_strategy,
                                                                     structure_environments=se,
                                                                     valences=valences,
                                                                     valences_origin=valences_origin)

        # Write to json file
        if 'json_file' in fw_spec:
            json_file = fw_spec['json_file']
        else:
            json_file = 'light_structure_environments.json'
        f = open(json_file, 'w')
        json.dump(lse.as_dict(), f)
        f.close()

        # Save to database
        if 'mongo_database' in fw_spec:
            database = fw_spec['mongo_database']
            entry = {'identifier': identifier,
                     'elements': [elmt.symbol for elmt in lse.structure.composition.elements],
                     'nelements': len(lse.structure.composition.elements),
                     'pretty_formula': lse.structure.composition.reduced_formula,
                     'nsites': len(lse.structure),
                     'chemenv_statistics': lse.get_statistics(statistics_fields='ALL', bson_compatible=True)
                     }
            saving_option = fw_spec['saving_option']
            if saving_option == 'gridfs':
                gridfs_msonables = {'structure': lse.structure,
                                    'light_structure_environments': lse}
            elif saving_option == 'storefile':
                gridfs_msonables = None
                if 'lse_prefix' in fw_spec:
                    lse_prefix = fw_spec['lse_prefix']
                    if not lse_prefix.isalpha():
                        raise ValueError('Prefix for light_structure_environments file is "{}" '
                                         'while it should be alphabetic'.format(lse_prefix))
                else:
                    lse_prefix = ''
                if lse_prefix:
                    lse_rfilename = '{}_{}.json'.format(lse_prefix, fw_spec['storefile_basename'])
                else:
                    lse_rfilename = '{}.json'.format(fw_spec['storefile_basename'])
                lse_rfilepath = '{}/{}'.format(fw_spec['storefile_dirpath'], lse_rfilename)
                storage_server = fw_spec['storage_server']
                storage_server.put(localpath=json_file, remotepath=lse_rfilepath, overwrite=True, makedirs=False)
                entry['light_structure_environments_file'] = lse_rfilepath
            else:
                raise ValueError('Saving option is "{}" while it should be '
                                 '"gridfs" or "storefile"'.format(saving_option))
            criteria = {'identifier': identifier}
            if database.collection.find(criteria).count() == 1:
                database.update_entry(query=criteria, entry_update=entry,
                                      gridfs_msonables=gridfs_msonables)
            else:
                database.insert_entry(entry=entry, gridfs_msonables=gridfs_msonables)