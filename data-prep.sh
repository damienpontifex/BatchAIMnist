#! /usr/bin/env bash

# -e: immediately exit if any command has a non-zero exit status
set -e

function finish {
	# Cleanup
	az configure --defaults group='' location=''
}
trap finish EXIT

RG_NAME=batch-rg
LOCATION=westus2
JOB_ID=$(date "+%Y-%m-%d-%H-%M-%S")

az configure --defaults group="$RG_NAME" location="$LOCATION"

export {AZURE_BATCHAI_STORAGE_ACCOUNT,AZURE_STORAGE_ACCOUNT}=pontifexml
export {AZURE_BATCHAI_STORAGE_KEY,AZURE_STORAGE_KEY}=$(az storage account keys list --account-name ${AZURE_STORAGE_ACCOUNT} --resource-group ml | head -n1 | awk '{print $3}')

az storage blob upload \
    -f data-prep/DataPreparation.py \
    -c machinelearning \
    -n mnist/data-prep/DataPreparation.py

az batchai job create \
    --config data-prep/data-prep-job.json \
    --name "mnist-data-prep-${JOB_ID}" \
    --cluster-name dsvm \
    --cluster-resource-group ${RG_NAME} \
    --resource-group ${RG_NAME}