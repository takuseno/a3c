echo "$$" > pid

while getopts e:n:l:r OPT
do
  case $OPT in
    "e" ) GYMENV_FLAG="TRUE" ; GYMENV="$OPTARG" ;;
    "n" ) NUM_FLAG="TRUE" ; NUM=$OPTARG ;;
    "l" ) LOGPATH_FLAG="TRUE" ; LOGPATH="$OPTARG" ;;
    "r" ) RENDER_FLAG="TRUE" ;;
  esac
done

if [ "$GYMENV_FLAG" != "TRUE" ]; then
  GYMENV="PongDeterministic-v4"
fi

if [ "$NUM_FLAG" != "TRUE" ]; then
  NUM=8
fi

if [ "$LOGPATH_FLAG" != "TRUE" ]; then
  LOGPATH="`date '+%y%m%d%H%M%S'`"
fi

for i in `seq 0 $(($NUM-1))`
do
  if [ "$RENDER_FLAG" == "TRUE" -a $i == 0 ]; then
    python main.py --num-processes $NUM --job worker --index $i --logdir $LOGPATH --env $GYMENV --render &
  else
    python main.py --num-processes $NUM --job worker --index $i --logdir $LOGPATH --env $GYMENV &
  fi
done
python main.py --num-processes $NUM --job ps --index 0 --logdir $LOGPATH
