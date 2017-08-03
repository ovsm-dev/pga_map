#!/bin/bash

# PROC name is first argument
PROC=$1

# If available, delay in minute is the second argument
# If not, default to 1 hour
if [ "$2" = "" ]
then delay="60"
else delay=$2
fi

# Load WEBOBS.rc variables
source /etc/webobs.d/../CODE/shells/readconf

# Set output path directory, create directory if it doesn't exists
outROOT="${WO__ROOT_OUTR}/${PROC}"
if [ ! -d ${outROOT} ]
then mkdir -p ${outROOT}
fi

# Load the PROC variables
oIFS=${IFS}; IFS=$'\n'
LEXP=($(gawk -v proc=${PROC} -F '|' '!/^(#|$|\r|=)/{gsub(/WEBOBS{/,"{WO__",$2);if(length($2)>1)printf("%s_%s=%s\n",proc,$1,$2)}' ${WO__PATH_PROCS}/${PROC}/${PROC}.conf))
for i in $(seq 0 1 $(( ${#LEXP[@]}-1 )) ); do export ${LEXP[$i]}; done
IFS=${oIFS}

# Input data is given by the rawdata path from the PROC
shakemapROOTvar=${PROC}_RAWDATA
if [[ ${!shakemapROOTvar} =~ .*\$\{* ]]
then  shakemapROOT=$(eval echo ${!shakemapROOTvar})
else  shakemapROOT=echo ${!shakemapROOTvar}
fi

# If the input directory doesn't exists, end of the program
if [ ! -d ${shakemapROOT} ]
then
	echo "!! Input directory ${shakemapROOT} not found !!"
	echo "!! Check config !!"
	exit 1
fi

# Search for files created or updated in the ${delay} last minutes
updated_files=$(find ${shakemapROOT} -type f -mmin -${delay} -iname event_dat.xml)

# Iterate over files and create graphs
for k in $(seq 0 $((${#updated_files[*]} - 1)))
do
	evt_file="$(dirname ${updated_files[$k]})/event.xml"
	dat_file=${updated_files[$k]}
	if [ -s ${evt_file} && -s ${dat_file} ]
	then python pga_map.py ${evt_file} ${dat_file} ${outROOT} -c ${WO__PATH_PROCS}/${PROC}/${PROC}.conf
	else
		echo "$(dirname ${updated_files[$k]})"
		echo "!! One of the input files is empty, skipping event !!"
	fi
done

exit 0

