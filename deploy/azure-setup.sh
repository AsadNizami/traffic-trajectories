#!/usr/bin/env bash
# One-time Azure setup: registry, Container App (scale to zero) and a GitHub OIDC identity for CI.
# Prereqs: az CLI logged in (`az login`), docker running. Run from the repo root.
set -euo pipefail

RG=${RG:-traffic-trajectories}
LOCATION=${LOCATION:-germanywestcentral}
ENV_NAME=${ENV_NAME:-traj-env}
APP=${APP:-traffic-trajectories}
REPO=${REPO:-AsadNizami/traffic-trajectories}

az extension add --name containerapp --upgrade -y
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ContainerRegistry; do
  az provider register -n "$ns" --wait
done

az group create -n "$RG" -l "$LOCATION" -o none
# Reuse the registry from an earlier run; otherwise pick a globally unique name
ACR=${ACR:-$(az acr list -g "$RG" --query "[0].name" -o tsv)}
ACR=${ACR:-trajacr$(openssl rand -hex 3)}
az acr show -n "$ACR" -o none 2>/dev/null || az acr create -g "$RG" -n "$ACR" --sku Basic -o none
# Express environments don't support managed-identity pulls, so the app pulls with the admin credentials
az acr update -n "$ACR" --admin-enabled true -o none

# First image is pushed from here so the app starts on a working revision
IMAGE="$ACR.azurecr.io/$APP:init"
az acr login -n "$ACR"
docker build -t "$IMAGE" .
docker push "$IMAGE"

az containerapp env show -g "$RG" -n "$ENV_NAME" -o none 2>/dev/null \
  || az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" -o none
az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" \
  --image "$IMAGE" --registry-server "$ACR.azurecr.io" --registry-username "$ACR" \
  --registry-password "$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)" \
  --ingress external --target-port 7860 \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1 -o none

# GitHub Actions identity: federated credential for pushes to main, Contributor on the resource group only
CLIENT_ID=$(az ad app create --display-name "$APP-github" --query appId -o tsv)
az ad sp create --id "$CLIENT_ID" -o none
az ad app federated-credential create --id "$CLIENT_ID" --parameters "{
  \"name\": \"github-main\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:$REPO:ref:refs/heads/main\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}" -o none
RG_ID=$(az group show -n "$RG" --query id -o tsv)
for _ in 1 2 3 4 5; do  # the new service principal can take a moment to replicate
  az role assignment create --assignee "$CLIENT_ID" --role Contributor --scope "$RG_ID" -o none && break
  sleep 15
done

URL=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
cat <<EOF

App: https://$URL

Set these GitHub repo variables so CI can deploy (none of them are secrets):
gh variable set AZURE_CLIENT_ID --body $CLIENT_ID
gh variable set AZURE_TENANT_ID --body $(az account show --query tenantId -o tsv)
gh variable set AZURE_SUBSCRIPTION_ID --body $(az account show --query id -o tsv)
gh variable set AZURE_RG --body $RG
gh variable set AZURE_ACR --body $ACR
gh variable set AZURE_APP --body $APP
EOF
