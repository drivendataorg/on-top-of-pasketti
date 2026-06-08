DIRBAK=$(pwd)
TGTDIR="../data/submission_${1:0:4}"
PROJ="csrw"


echo $TGTDIR
echo $1

# prepare
rm -rf $TGTDIR/*
mkdir -p $TGTDIR/$PROJ/data
mkdir -p $TGTDIR/$PROJ/$PROJ

# code
CODEDIR="$TGTDIR/$PROJ/$PROJ"
cp main.py $TGTDIR
cp main.py $CODEDIR
cp util.py $CODEDIR
cp dataset.py $CODEDIR
cp sft.py $CODEDIR
cp pt_util.py $CODEDIR
cp score_func.py $CODEDIR
cp -r -L ../qwen_asr $TGTDIR/$PROJ/
#cp -r -L ../data/${1:0:4}_pkg $TGTDIR/$PROJ/pkg
if [[ "$1" == *"cohere"* ]]; then
    cp -r -L ../data/cohere_pkg $TGTDIR/$PROJ/pkg
fi

# lora
#mkdir -p $TGTDIR/$PROJ/data/hfmodels/unsloth
#python fix_lora.py "$1"

# model
cd $TGTDIR/$PROJ/data
for modelname in $1
do
  ln -s "../../../$modelname" "$modelname"
done

# pack
cd $DIRBAK
cd $TGTDIR
zip -r submission.zip ./* -x ./$PROJ/data/*/events* -x ./$PROJ/data/*/pred*

cd $DIRBAK
