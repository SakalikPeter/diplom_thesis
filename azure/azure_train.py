import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.environment import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
from azureml.data.datapath import DataPath
from azureml.core.authentication import InteractiveLoginAuthentication


ws = Workspace.from_config('azure/azure_config.json', auth=InteractiveLoginAuthentication())
dataset = Dataset.get_by_name(workspace=ws, name='MoNuSAC')

config = ScriptRunConfig(source_directory='.',
                         script='stylegan.py',
                         arguments=['--resolution', '256',
                                    '--wgan_penalty_const', '10',
                                    '--discriminator_iterations', '5',
                                    '--iterations', '15000',
                                    '--interval', '1000',
                                    '--max_log2res', '8',
                                    '--grow_model', '0',
                                    '--wandb_key', '',
                                    '--data_path', dataset.as_mount()], # This is important how to mount dataset from DataStore
                         compute_target='Test-cluster') # Compute target is your created compute cluster


experiment = Experiment(workspace=ws, name='sakalik-test')

azureml.core.runconfig.DockerConfiguration(use_docker=True)
env = Environment.from_pip_requirements(name='tf-gpu-sakalik', file_path='requirements.txt')
env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

env.register(workspace=ws)
env = Environment.get(workspace=ws, name='tf-gpu-sakalik')


config.run_config.environment = env
run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)