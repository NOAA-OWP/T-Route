# T-Route FastAPI

The following doc is meant to explain the T-Route FastAPI implementation using docker compose and shared volumes. 

## Why an API?

T-Route is used in many contexts for hydrological river routing:
- NGEN 
- Scientific Python Projects
- Replace and Route (RnR)

In the latest PR for RnR (https://github.com/NOAA-OWP/hydrovis/pull/865), there is an requirement to run T-Route as a service. This service requires an easy way to dynamically create config files, restart flow from Initial Conditions, and run T-Route. To satisfy this requirement, a FastAPI endpoint was created in `/src/app` along with code to dynamically create t-route endpoints. 

## Why use shared volumes?

Since T-Route is running in a docker container, there has to be a connection between the directories on your machine and the directories within the container. We're sharing the following folders by default:
- `data/rfc_channel_forcings`
  - For storing RnR RFC channel domain forcing files (T-Route inputs)
- `data/rfc_geopackage_data`
  - For storing HYFeatures gpkg files 
  - Indexed by the NHD COMID, also called hf_id. Ex: 2930769 is the hf_id for the CAGM7 RFC forecast point. 
- `data/troute_restart`
  - For storing TRoute Restart files
- `data/troute_output`
  - For outputting results from the T-Route container

## Quickstart
1. From the Root directory, run:
```shell
docker compose --env-file docker/test_troute_api.env -f docker/compose.yaml up --build
```

This will start the T-Route container and run the API on localhost:8004. To view the API spec and swagger docs, visit localhost:8004/docs

2. Submit a request
```shell
curl -X 'GET' \
  'http://localhost:8004/api/v1/flow_routing/v4/?lid=CAGM7&feature_id=2930769&hy_id=1074884&initial_start=0&start_time=2024-08-24T00%3A00%3A00&num_forecast_days=5' \
  -H 'accept: application/json'
```

This curl command is pinging the flow_routing v4 endpoint `api/v1/flow_routing/v4/` with the following metadata:
```
lid=CAGM7
feature_id=2930769
hy_id=1074884
initial_start=0
start_time=2024-08-24T00:00:00
num_forecast_days=5
```

which informs T-Route which location folder to look at, what feature ID to read a gpkg from, the HY feature_id where flow is starting, the initial start flow for flow restart, start-time of the routing run, and the number of days to forecast. 

You can also run the following args from the swagger endpoint:

![alt text](swagger_endpoints.png)

The result for a successful routing is a status 200:
```json
{
  "message": "T-Route run successfully",
  "lid": "CAGM7",
  "feature_id": "2930769"
}
```

and an internal 500 error if there is something wrong. 

## Building and pushing to a container registry

To ensure Replace and Route is using the correct version of T-Route, it is recommended a docker container be built, and then pushed to a registry (Dockerhub, GitHub Container Registry, etc). To do this manually for the GitHub container registry, the following commands should be used within a terminal.

```shell
docker login --username ${GH_USERNAME} --password ${GH_TOKEN} ghcr.io
```
- This command will log the user into the GitHub container registry using their credentials

```shell
docker build -t ghcr.io/NOAA-OWP/t-route/t-route-api:${TAG} -f docker/Dockerfile.troute_api
```
- This command builds the T-Route API container using a defined version `${TAG}`

```shell
docker push ghcr.io/NOAA-OWP/t-route/t-route-api:${TAG}
```
- This commands pushes the built T-Route API container to the NOAA-OWP/t-route container registry


The following env variables are used:
- `${GH_USERNAME}`
  - your github username
- `${GH_TOKEN}`
  - your github access token
- `${TAG} `
  - the version tag
  - ex: 0.0.2

If you want to build this off a forked version, change the container registry (`/NOAA-OWP/t-route/`) to your user accounts container registry.

## Testing:

See `test/api/README.md` for testing information
