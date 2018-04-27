#!/bin/bash

# -e: immediately exit if any command has a non-zero exit status
set -e

function finish {
	# Cleanup
	az configure --defaults group='' location=''
}
trap finish EXIT

RG_NAME=batch-rg
LOCATION=westus2

az configure --defaults group="$RG_NAME" location="$LOCATION"

# Resource Group
if [ $(az group exists -n "$RG_NAME") == "false" ]; then
    az group create -n "$RG_NAME"
fi

export {AZURE_BATCHAI_STORAGE_ACCOUNT,AZURE_STORAGE_ACCOUNT}=pontifexml
export {AZURE_BATCHAI_STORAGE_KEY,AZURE_STORAGE_KEY}=$(az storage account keys list --account-name ${AZURE_STORAGE_ACCOUNT} --resource-group ml | head -n1 | awk '{print $3}')

# Batch AI Cluster
az batchai cluster create \
    --name dsvm \
    --image UbuntuDSVM \
    --bfs-name machinelearning \
    --vm-size Standard_NC6 \
    --min 0 --max 3 \
    --user-name $USER --ssh-key ~/.ssh/id_rsa.pub \
    -c cluster/clusterconfig.json
