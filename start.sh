git clone https://github.com/marynadorosh/test_task.git

wget --quiet https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz
cp -a  cc.en.300.vec /data/interim/

mkdir processed/train/
mkdir processed/test/
