### Strip excess files and zip 
VIRTUAL_ENV=$1

find $VIRTUAL_ENV -name "*.dist-info" -type d -prune -exec rm -rf {} \;
echo "venv stripped size $(du -sh $VIRTUAL_ENV | cut -f1)"

rm -rf $VIRTUAL_ENV/lib64/python3.6/site-packages/matplotlib/mpl-data/sample_data

tar -cvf "$VIRTUAL_ENV/lib64/python3.6/site-packages/astropy.tar" "$VIRTUAL_ENV/lib64/python3.6/site-packages/astropy"
rm -rf "$VIRTUAL_ENV/lib64/python3.6/site-packages/astropy"

# Clean up tests
find $VIRTUAL_ENV -name "tests" -type d -prune -exec rm -rf {} \;
echo "venv stripped size $(du -sh $VIRTUAL_ENV | cut -f1)"

cd /
tar -xvf "$VIRTUAL_ENV/lib64/python3.6/site-packages/astropy.tar"
rm -rf "$VIRTUAL_ENV/lib64/python3.6/site-packages/astropy.tar"

# Astropy test data
files=`find $VIRTUAL_ENV/lib64/python3.6/site-packages/astropy -name "data" -type d -prune | grep test`
rm -rf $files

cd ${VIRTUAL_ENV}/lib64/python3.6/site-packages/scipy/.libs
ln -sf ../../numpy/.libs/libopenblasp-r0-8dca6697.3.0.dev.so libopenblasp-r0-39a31c03.2.18.so 

echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
find $VIRTUAL_ENV/lib64/python3.6/site-packages/ -name "*.so" | xargs strip
echo "venv stripped size $(du -sh $VIRTUAL_ENV | cut -f1)"

# Clean up cache
echo "venv stripped size $(du -sh $VIRTUAL_ENV | cut -f1)"
find $VIRTUAL_ENV -name "__pycache__" -type d -prune -exec rm -rf {} \;
echo "venv stripped size $(du -sh $VIRTUAL_ENV | cut -f1)"

# Make zip file
cp /workdir/*.py $VIRTUAL_ENV
pushd $VIRTUAL_ENV && zip -r -9 -q /tmp/process.zip *.py; popd

pushd $VIRTUAL_ENV/lib/python3.6/site-packages/ && zip -r -9 --out /tmp/partial-venv.zip -q /tmp/process.zip * ; popd
pushd $VIRTUAL_ENV/lib64/python3.6/site-packages/ && zip -r -9 --out /tmp/venv.zip -q /tmp/partial-venv.zip * ; popd
echo "site-packages compressed size $(du -sh /tmp/venv.zip | cut -f1)"

cp /tmp/venv.zip /workdir/
