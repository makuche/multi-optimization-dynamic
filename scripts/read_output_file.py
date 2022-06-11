import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from pathlib import Path


def parse_values(line, typecast=int, sep=None, idx=1, cut_idx=None):
    return [typecast(val.strip(sep)) for val in line.split(sep)[idx:cut_idx]]


class OutputFileParser:
    def __init__(self, file_name, folder, hf_cost=1, artificial_cost=None):
        self.file_name = file_name
        self.folder_file_path = Path(__file__).resolve().parent
        self.out_file_path = self.folder_file_path / f"{folder}/{file_name}.out"
        self.rst_file_path = self.folder_file_path / f"{folder}/{file_name}.rst"
        self.data = defaultdict(list)
        self.hf_cost = hf_cost
        self.artificial_cost = artificial_cost
        self.read_data()
        self.calculate_cumulative_cost()

    def read_data(self):
        xy = []
        acq_times = []
        best_acq = []
        global_min_prediction = []
        global_min_prediction_convergence = []
        gp_hyperparam = []
        iter_times = []
        total_time = []
        boss_version = None
        with open(self.out_file_path, "r") as file:
            lines = file.readlines()
            self.data["header"] = lines[0:100]
            for i in range(len(lines)):
                line = lines[i]
                if "Version" in line:
                    boss_version = parse_values(line, typecast=str)[0]
                if "Data point added to dataset" in line:
                    line = lines[i + 1]
                    xy.append(parse_values(line, typecast=float, idx=0))
                elif "Best acquisition" in line:
                    line = lines[i + 1]
                    best_acq.append(parse_values(line, typecast=float, idx=0))
                elif "Global minimum prediction" in line:
                    line = lines[i + 1]
                    global_min_prediction.append(
                        parse_values(line, typecast=float, idx=0)
                    )
                elif "Global minimum prediction" in line:
                    line = lines[i + 1]
                    global_min_prediction_convergence.append(
                        parse_values(line, typecast=float, idx=0)
                    )
                elif "GP model hyperparameters" in line:
                    line = lines[i + 1]
                    gp_hyperparam.append(parse_values(line, typecast=float, idx=0))
                elif "Iteration time [s]:" in line:
                    # If line contains str and float types, casting to str and then
                    # manually to float again has to be done
                    iter_times.append(float(parse_values(line, typecast=str, idx=3)[0]))
                    # Here not needed because line only contains a float
                    total_time.append(parse_values(line, typecast=float, idx=7)[0])
                elif "Objective function evaluated" in line:
                    acq_times.append(parse_values(line, typecast=float, idx=6)[0])
                elif ("initpts" in line) and ("iterpts" in line):
                    self.data["initpts"] = parse_values(line, cut_idx=-2)[0]
                    self.data["iterpts"] = parse_values(line, idx=3)
                elif "inittype" in line:
                    self.data["inittype"] = parse_values(line, typecast=str)
                    self.data["num_tasks"] = len(self.data["inittype"])
                elif "bounds" in line and len(self.data["bounds"]) == 0:
                    tmp = " ".join(parse_values(line, typecast=str, idx=1))
                    self.data["bounds"] = np.array(
                        parse_values(tmp, typecast=str, sep=";", idx=0)
                    )
                # elif 'kernel' in line:
                #     self.data['kernel'] = parse_values(line, typecast=str, idx=1)
                elif "kerntype" in line:
                    self.data["kernel"] = parse_values(line, typecast=str)
                elif "yrange" in line:
                    self.data["yrange"] = np.array(parse_values(line, typecast=str))
                elif "thetainit" in line:
                    self.data["thetainit"] = parse_values(line, typecast=str)
                elif "thetapriorpar" in line:
                    tmp = " ".join(parse_values(line, typecast=str))
                    self.data["thetapriorpar"] = parse_values(
                        tmp, typecast=str, sep=";", idx=0
                    )
                elif "|| Bayesian optimization completed" in line:
                    self.data["run_completed"] = True

        self.data["xy"] = np.array(xy)
        self.data["dim"] = len(xy[0]) - 1
        self.data["acq_times"] = np.array(acq_times)
        self.data["best_acq"] = np.array(best_acq)
        self.data["gmp"] = np.array(global_min_prediction)
        self.data["gmp_convergence"] = global_min_prediction_convergence
        self.data["GP_hyperparam"] = gp_hyperparam
        self.data["iter_times"] = np.array(iter_times)
        self.data["total_time"] = np.array(total_time)
        if "run_completed" not in self.data:
            self.data["run_completed"] = False

        # Get sample indices from the rst file
        with open(self.rst_file_path, "r") as f:
            start_reading_indices = False
            for line in f:
                if line.startswith("acqcost"):
                    if "acqcost_as_timing" in line:
                        continue
                    if "None" in line:
                        continue
                    self.data["acqcost"] = parse_values(
                        line, typecast=float, sep=" ", idx=1
                    )
                if line.startswith("RESULTS"):
                    start_reading_indices = True
                    continue
                if start_reading_indices:
                    self.data["sample_indices"].append(
                        int(float(line.split()[self.data["dim"]])))

    def calculate_cumulative_cost(self):
        costs = self.data["acqcost"] if self.artificial_cost is None \
            else self.artificial_cost
        if self.data["num_tasks"] > 1:
            iter_cost = self._get_iter_cost()
            self.data["cumulative_cost"] = np.cumsum(iter_cost)
        else:
            self.data["cumulative_cost"] = self.hf_cost * np.arange(
                1, len(self.data["xy"]) + 1
            )

    def _get_iter_cost(self):
        iter_cost = []
        costs = self.data["acqcost"] if self.artificial_cost is None \
            else self.artificial_cost
        for idx in self.data["sample_indices"]:
            for task_idx in range(self.data["num_tasks"]):
                if task_idx == idx:
                    iter_cost.append(costs[task_idx])
        return iter_cost


class ParserToDataFrame:
    """For each of the parser objects, determine the

    DataFrame has columns:
    fidelities, acqfn, strategy, dim, num_tasks, sample_indices, acqcosts, cumulative_costs,
    cost_to_convergence_0_1
    """

    def __init__(self, parser_objects, tolerance=0.1, true_min=-202861.3237):
        if not isinstance(parser_objects, list):
            parser_objects = [parser_objects]
        self.parser_objects = parser_objects
        self.tolerance = tolerance
        self.true_min = true_min
        self.df = pd.DataFrame()
        self.parsed_objects_to_dataframe()
        self.create_dataframe()

    def __call__(self):
        return self.df

    def parsed_objects_to_dataframe(self):
        """ """
        self.fidelities = []
        self.dims = []
        self.acqfns = []
        self.strategies = []
        self.run_indices = []
        self.num_tasks = []
        self.sample_indices = []
        self.acqcosts = []
        self.acqtimes = []
        self.iteration_times = []
        self.cumulative_costs = []
        self.convergence_idx = []
        self.convergence_cost = []
        for obj in self.parser_objects:
            name = str(obj.out_file_path).split("/")[-1].split(".")[0]
            if ("strategy" in name) or ("inseparable" in name):
                fidelity_cut_idx = 1
                self.fidelities.append(name.split("_")[0] + "_" + name.split("_")[1])
            else:
                fidelity_cut_idx = 0
                self.fidelities.append(name.split("_")[0])
            self.dims.append(name.split("_")[fidelity_cut_idx + 1])
            self.acqfns.append(name.split("_")[fidelity_cut_idx + 2])
            if self.acqfns[-1] == "mumbo":
                self.strategies.append("strategy6")
            else:
                self.strategies.append(name.split("_")[fidelity_cut_idx + 3])
            self.run_indices.append(name.split("_")[fidelity_cut_idx + 4])
            initpts = obj.data["initpts"]
            self.num_tasks.append(obj.data["num_tasks"])
            self.sample_indices.append(obj.data["sample_indices"])
            self.acqcosts.append(obj.data["acqcost"])
            self.acqtimes.append(obj.data["acq_times"])
            self.iteration_times.append(obj.data["iter_times"])
            self.cumulative_costs.append(
                obj.data["cumulative_cost"][initpts - 1 :])
            yhat_predicts = obj.data["gmp"][:, -2]
            convergence_idx = self.get_convergence_index(
                yhat_predicts - self.true_min, self.tolerance
            )
            self.convergence_idx.append(convergence_idx)
            if convergence_idx is None:
                self.convergence_cost.append(None)
            else:
                self.convergence_cost.append(
                    self.cumulative_costs[-1][convergence_idx]
                )

    def get_convergence_index(self, y, tolerance):
        mask = np.argwhere(np.abs(y) > tolerance)
        if mask[-1] + 1 == len(y):
            return None
        else:
            indices_within_tolerance = mask[-1] + 1
            return int(indices_within_tolerance[0])

    def create_dataframe(self):
        self.df = pd.DataFrame(
            {
                "fidelity": self.fidelities,
                "dim": self.dims,
                "acqfn": self.acqfns,
                "strategy": self.strategies,
                "run_index": self.run_indices,
                "num_tasks": self.num_tasks,
                "sample_indices": self.sample_indices,
                "acqcosts": self.acqcosts,
                "acquisition times": self.acqtimes,
                "iteration times": self.iteration_times,
                "cumulative_costs": self.cumulative_costs,
                "convergence_idx": self.convergence_idx,
                "convergence_cost": self.convergence_cost,
            }
        )
