# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md We're going to rerun the previous notebook which imported our data, preprocessed, then registered the model.

# COMMAND ----------

dbutils.notebook.run('01 - MLflow Model Life Cycle', timeout_seconds=120)

# COMMAND ----------

# MAGIC %md 
# MAGIC Now we are going to find the Run ID of the model programmatically

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "ms course"').iloc[0].run_id

# COMMAND ----------

# MAGIC %md Let's store the model version here. Since we have already logged a few versions of the model, we'll get a notification along with our most current version number.

# COMMAND ----------

model_version = mlflow.register_model(f"runs:/{run_id}/sklearn-model", model_name)

# COMMAND ----------

# MAGIC %md Now let's transition this model to production and load it into this notebook

# COMMAND ----------

from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Staging",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC The Models page now shows the model version in stage "Production". 
# MAGIC 
# MAGIC Let's load the model back into this notebook from the registry:

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/staging")

# COMMAND ----------

# MAGIC %md Now we would like to use this model loaded from the MLflow registry to get predictions in our notebook.

# COMMAND ----------

model.predict(X_test)

# COMMAND ----------

