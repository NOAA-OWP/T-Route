## Testing:

### Set up Using a Dockerfile:

To build the T-Route api from a dockerfile, you can use the following commands from the t-route root repo:

```shell
docker build -t troute_api -f docker/Dockerfile.troute_api .
```

```shell
docker run -p 8000:8000   \
--env-file docker/test_troute_api.env  \
-v ${OUTPUT_VOLUME_SOURCE}:${OUTPUT_VOLUME_TARGET}  \
-v ${DATA_VOLUME_SOURCE}:${DATA_VOLUME_TARGET}  \
-v ${CORE_VOLUME_SOURCE}:${CORE_VOLUME_TARGET}  \
-v ${TEST_SOURCE}:${TEST_TARGET}   troute_api
```

### Set up Docker Compose:

Docker Compose uses a YAML file to configure docker containers and execution. To install compose, you can follow the examples on docker's docs: https://docs.docker.com/compose/install/linux/

To run compose, you can use the following command from the root directory:

```shell
docker compose --env-file docker/test_troute_api.env -f docker/compose.yaml up --build
```

if you want to build a compose container to mimic what is used in RnR, you can run the following steps
```shell
docker compose --env-file docker/rnr_compose.env -f docker/compose.yaml up --build
```

#### Testing the RnR endpoint in the API: 
The following folder contains data files that are to be used to test the T-Route FastAPI code within src/app

1. Follow the steps above to build/run either the docker container, or docker compose
2. visit `localhost:8000/docs` in your browser
3. Enter the following parameters into the `/api/v1/flow_routing/v4` endpoint
- lid=CAGM7
- feature_id=2930769
- hy_id=1074884
- initial_start=0
- start_time=2024-08-24T00:00:00
- num_forecast_days=5
4. Click execute
5. A Status 201 code means the run ran, and test/api/data/troute_output will be populated in the `{lid}/` folder

#### Testing the LowerColorado test cases through docker compose:
1. Follow the steps above to build/run either the docker container, or docker compose
2. visit `localhost:8000/docs` in your browser
3. Execute the `/api/v1/flow_routing/v4/tests/LowerColorado` endpoint using the default parameter file path for LowerColorado_TX_v4 
4. A Status 201 code means the run ran, and the defined yaml output will be populated
