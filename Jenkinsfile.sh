set -e
set -u

WORKSPACE_ROOT="$WORKSPACE/$BUILD_NUMBER"
TEST_OUTPUT_DIR="$WORKSPACE_ROOT/test-output"
TEST_DATA_DIR="$WORKSPACE/../../../test-data"
mkdir $TEST_OUTPUT_DIR

# build and testrun
docker build -f ${WORKSPACE_ROOT}/projects/Cubical/.jenkins/1804.py3.docker -t cubical.1804.py3:${BUILD_NUMBER} ${WORKSPACE_ROOT}/projects/Cubical/
docker run --rm cubical.1804.py3:${BUILD_NUMBER}

#run tests

img=1804.py3

docker run --rm -m 100g --cap-add sys_ptrace \
                    --memory-swap=-1 \
                    --shm-size=150g \
                    --rm=true \
                    --name=cubical$BUILD_NUMBER \
                    -v ${TEST_OUTPUT_DIR}:/workspace \
                    -v ${TEST_OUTPUT_DIR}:/root/tmp \
                    --entrypoint /bin/bash \
                    cubical.${img}:${BUILD_NUMBER} \
                    -c "cd /src/cubical && nosetests --with-xunit --xunit-file /workspace/nosetests.xml test"

