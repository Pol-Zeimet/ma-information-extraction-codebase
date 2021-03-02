from datetime import datetime
from typing import List
import mlflow
import sys
import traceback
from src.experiments.experiment_class.experiment_class import BaseExperiment
import os
import gpustat
import time


class ExperimentPipeline:
    def run(self, experiments: List[BaseExperiment]):
        index = -1
        while index < 0:
            time.sleep(1)
            index = self.gpu_stat_wait_until_free()
        print(f"running on GPU {index}")
        for i, experiment in enumerate(experiments):
            try:
                print(
                    f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}  Experiment {i + 1}/{len(experiments)}: {experiment.name}")
                experiment.setup()
                experiment.run()

            except:
                filename = "ma-information-extraction-codebase/graph_networks/logs/error_log.txt"
                print(f'An error occurred while executing experiment {experiment.name}')
                print(f"writing to file {filename}")
                with open(os.path.join(os.getcwd(), filename), 'a') as f:
                    f.write(datetime.today().strftime('%d-%m-%Y-%H:%M:%S'))
                    f.write('--------------------------------------------------------------------------------')
                    f.write('\n')
                    f.write(f'Error in Experiment run for {experiment.name} \n')
                    f.write(str(sys.exc_info()))
                    f.write('\n')
                    traceback.print_exc(file=f)
                    f.write('\n')
                    f.write('--------------------------------------------------------------------------------')
                    f.write('\n')
                    f.close()
                print('Trying to execute remaining experiment_scripts')

                if mlflow.active_run():
                    mlflow.end_run()

            finally:
                experiment.cleanup()
                print("\n")

    @staticmethod
    def gpu_stat_wait_until_free(free_memory: int = 10000):
        gpustats = gpustat.GPUStatCollection.new_query()
        g_json = gpustats.jsonify()
        for i, g in enumerate(g_json["gpus"]):
            if (g["memory.total"] - g["memory.used"] > free_memory):
                os.environ['CUDA_VISIBLE_DEVICES'] = f'{i}'
                return i
            else:
                # print("fucking fully loaded GPU")
                pass
        return -1

