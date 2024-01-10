import mlflow
from mlflow import MlflowClient

client = MlflowClient()
runs = client.search_runs("0")
run_valid_losses = [run.data.metrics["valid_loss"] for run in runs]
minv = min(run_valid_losses)
run_loc = run_valid_losses.index(minv)
run = runs[run_loc]

metric_history = client.get_metric_history(run.info.run_id, key="valid_loss")
step_valid_losses = [metric.value for metric in metric_history]
minv = min(step_valid_losses)
step_loc = step_valid_losses.index(minv)
step = metric_history[step_loc].step
# client.list_artifacts(run.info.run_id)
# import pdb; pdb.set_trace()