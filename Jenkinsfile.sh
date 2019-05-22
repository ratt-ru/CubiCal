WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"
TEST_OUTPUT_DIR="$WORKSPACE_ROOT/test-output"
TEST_DATA_DIR="$WORKSPACE/../../../test-data"
mkdir $TEST_OUTPUT_DIR

# build and testrun
docker build -t cubical:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/Cubical/
docker run --rm cubical:${BUILD_NUMBER}

#run tests
docker run --rm -m 100g --cap-add sys_ptrace \
				   --memory-swap=-1 \
                   --shm-size=150g \
                   --rm=true \
                   --name=cubical$BUILD_NUMBER \
                   -v ${TEST_OUTPUT_DIR}:/workspace \
                   -v ${TEST_OUTPUT_DIR}:/root/tmp \
                   --entrypoint /bin/bash \
                   cubical:${BUILD_NUMBER} \
                   -c "cd /src/cubical && apt-get install -y git && pip install -r requirements.test.txt && nosetests --with-xunit --xunit-file /workspace/nosetests.xml test"