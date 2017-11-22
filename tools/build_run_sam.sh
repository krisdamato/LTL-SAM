rm -r /home/krisdamato/sam-build
mkdir /home/krisdamato/sam-build
cp /home/krisdamato/copy_sam_source.sh /home/krisdamato/sam-build
cd /home/krisdamato/sam-build
./copy_sam_source.sh
cd ../LTL-SAM/sam
python3 main.py
