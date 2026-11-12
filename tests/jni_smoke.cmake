# Host-JVM test of the JNI bridge (ctest jni_smoke): compile the Java mirror of the Kotlin natives and the test, run it
# with -Xcheck:jni against the built libforge_jni.
#   cmake -DJAVAC=.. -DJAVA=.. -DSRC=<tests/jni> -DOUT=<work dir> -DLIBDIR=<dir of libforge_jni> -DFPK=.. -DBERT=.. -P jni_smoke.cmake
file(MAKE_DIRECTORY ${OUT})
execute_process(COMMAND ${JAVAC} -Xlint:all -Werror -d ${OUT} ${SRC}/dev/forge/ForgeNative.java ${SRC}/JniSmoke.java RESULT_VARIABLE rc)
if(rc)
  message(FATAL_ERROR "jni_smoke: javac failed")
endif()
execute_process(COMMAND ${JAVA} -Xcheck:jni -Djava.library.path=${LIBDIR} -cp ${OUT} JniSmoke ${FPK} ${BERT} ${OUT}
                RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("${out}${err}")
if(rc)
  message(FATAL_ERROR "jni_smoke: failed (${rc})")
endif()
# -Xcheck:jni reports misuse as warnings on stdout / stderr and carries on
if("${out}${err}" MATCHES "WARNING in native method|FATAL ERROR in native method")
  message(FATAL_ERROR "jni_smoke: -Xcheck:jni reported a problem")
endif()
