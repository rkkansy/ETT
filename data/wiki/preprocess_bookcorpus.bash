set -e

lg=$1

# data path
BOOK_PATH=data/bookcorpus-cased
MAIN_PATH=$BOOK_PATH

# tools paths
TOOLS_PATH=$MAIN_PATH/tools
TOKENIZE=$TOOLS_PATH/tokenize.sh
REMOVE_ACCENT=$TOOLS_PATH/remove_accent.py

data/wiki/install-tools.sh $TOOLS_PATH

mkdir -p $BOOK_PATH/txt

python $MAIN_PATH/download_bookcorpus.py $BOOK_PATH/txt

INPUT_FILE=$BOOK_PATH/txt/bookcorpus_train.txt
OUTPUT_FILE=$BOOK_PATH/txt/train.raw
if [ -f $INPUT_FILE ]; then
    echo "*** Cleaning and tokenizing BookCorpus train data ... ***"
    cat $INPUT_FILE \
    | sed "/^\s*\$/d" \
    | $TOKENIZE $lg $TOOLS_PATH \
    | python $REMOVE_ACCENT \
    > $OUTPUT_FILE
    echo "*** Tokenized (+ accent-removal) BookCorpus data to $OUTPUT_FILE ***"
fi

# Cleanup
mv $BOOK_PATH/txt/* $BOOK_PATH/
rm -rf $BOOK_PATH/txt