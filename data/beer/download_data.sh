echo "Downloading test data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/annotations.json
echo "Downloading word embeddings"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/review+wiki.filtered.200.txt.gz
echo "Downloading combined train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.260k.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.260k.train.txt.gz
echo "Downloading aspect 0 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect0.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect0.train.txt.gz
echo "Downloading aspect 1 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect1.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect1.train.txt.gz
echo "Downloading aspect 2 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect2.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect2.train.txt.gz
gunzip --keep -v reviews.*.txt.gz