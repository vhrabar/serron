# Writes INPUT (a fatbin) to OUTPUT as a C++ header holding it in a `static const unsigned long long NAME[]`.
# The 8-byte element type gives the alignment cudaLibraryLoadData expects of an in-memory image.
#
#   cmake -DBIN2C=<bin2c> -DINPUT=<fatbin> -DOUTPUT=<header> -DNAME=<array name> -P embed_fatbin.cmake

foreach(_var IN ITEMS BIN2C INPUT OUTPUT NAME)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "embed_fatbin.cmake: ${_var} is not set")
    endif()
endforeach()

get_filename_component(_dir "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_dir}")

execute_process(
        COMMAND "${BIN2C}" --name "${NAME}" --type longlong --const --static "${INPUT}"
        OUTPUT_FILE "${OUTPUT}"
        RESULT_VARIABLE _result
)
if(NOT _result EQUAL 0)
    file(REMOVE "${OUTPUT}")
    message(FATAL_ERROR "bin2c failed on ${INPUT} (exit ${_result})")
endif()
