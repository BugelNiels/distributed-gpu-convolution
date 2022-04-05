read -p 'Job file name: ' -r outputfile
read -p 'Input image directory: ' -r imDir

filenames=`ls $imDir*.pgm` || exit 1
# empty file and write length to the file
echo "$filenames" | wc -w > $outputfile

maxWidth=0
maxHeight=0
for path in $filenames
do
    nums=$(head -3 $path)
		stringArray=($nums)
		curWidth=${stringArray[1]}
		curHeight=${stringArray[2]}
		maxWidth=$(( $curWidth > $maxWidth ? $curWidth : $maxWidth ))
		maxHeight=$(( $curHeight > $maxHeight ? $curHeight : $maxHeight ))
done

read -p 'Dynamic range (in bits): ' numBits
echo $numBits >> $outputfile

echo $maxWidth $maxHeight >> $outputfile

read -p 'Number of pixels padded horizontally: ' padX
read -p 'Number of pixels padded vertically: ' padY
echo $padX $padY >> $outputfile

#write each file to the list
for eachfile in $filenames
do
   echo $eachfile >> $outputfile
done