# cmake -DINPUT=file -DOUTPUT=file.inc -P embed_text.cmake: the file as a C++ raw string literal.
file(READ "${INPUT}" content)
file(WRITE "${OUTPUT}" "R\"VKEMBED(${content})VKEMBED\"\n")
