echo "Download Hotel Reviews"
wget -q --show-progress https://people.csail.mit.edu/yujia/files/r2a/data.zip
unzip data.zip
mkdir annotations
cd data || exit
echo "Copying Data"
cp oracle/hotel_*.train ../ -v
cp oracle/hotel_*.dev ../ -v
cp target/hotel_*.train ../annotations -v
cd ../
rm -rf data
rm data.zip