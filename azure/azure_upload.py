from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.datastore import Datastore
from glob import glob


ws = Workspace.from_config(path='azure/azure_config.json', auth=InteractiveLoginAuthentication())
datastore = Datastore.get_default(ws)

models = glob('saved_models/*/*.pb')
models2 = glob('saved_models/*/*/*')
files = models + models2

datastore.upload_files(files=files, relative_root='.', target_path='datasets/MoNuSAC', overwrite=True)