echo "$$" > pid

while getopts e:n:l:rm: OPT
do
  case $OPT in
    "e" ) GYMENV_FLAG="TRUE" ; GYMENV="$OPTARG" ;;
    "n" ) NUM_FLAG="TRUE" ; NUM=$OPTARG ;;
    "l" ) LOGPATH_FLAG="TRUE" ; LOGPATH="$OPTARG" ;;
    "r" ) RENDER_FLAG="TRUE" ;;
    "m" ) MODELPATH_FLAG="TRUE" ; MODELPATH="$OPTARG" ;;
  esac
done

ARGS=""

if [ "$GYMENV_FLAG" != "TRUE" ]; then
  GYMENV="PongDeterministic-v4"
fi
ARGS="$ARGS --env $GYMENV"

if [ "$NUM_FLAG" != "TRUE" ]; then
  NUM=8
fi
ARGS="$ARGS --num-processes $NUM"

if [ "$LOGPATH_FLAG" != "TRUE" ]; then
  LOGPATH="`date '+%y%m%d%H%M%S'`"
fi
ARGS="$ARGS --logdir $LOGPATH"

if [ "$MODELPATH_FLAG" == "TRUE" ]; then
  ARGS="$ARGS --load $MODELPATH"
fi

for i in `seq 0 $(($NUM-1))`
do
  TMPARGS="$ARGS"
  if [ "$RENDER_FLAG" == "TRUE" -a $i == 0 ]; then
    TMPARGS="$ARGS --render"
  fi
  python main.py --job worker --index $i $TMPARGS &
done
python main.py --job ps --index 0 $ARGS
