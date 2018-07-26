LOGPATH="`date '+%y%m%d%H%M%S'`"

while getopts e:n: OPT
do
  case $OPT in
    "e" ) GYMENV_FLAG="TRUE" ; GYMENV="$OPTARG" ;;
    "n" ) NUM_FLAG="TRUE" ; NUM=$OPTARG ;;
  esac
done

if [ "$GYMENV_FLAG" != "TRUE" ]; then
  GYMENV="PongDeterministic-v4"
fi

if [ "$NUM_FLAG" != "TRUE" ]; then
  NUM=8
fi

for i in `seq 0 $(($NUM-1))`
do
  python main.py --num-processes $NUM --job worker --index $i --logdir $LOGPATH --env $GYMENV &
done
python main.py --num-processes $NUM --job ps --index 0 --logdir $LOGPATH