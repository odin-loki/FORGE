/* C consumer of an installed FORGE: forge.h + libforge through the forge::c target. */
#include <forge.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    if (forge_abi_version() != FORGE_ABI_VERSION) { puts("ABI version mismatch between forge.h and libforge"); return 1; }
    const char* v = forge_version();
    if (!v || !v[0]) { puts("empty forge_version()"); return 2; }
    if (forge_open("/nonexistent/model.fpk", 0, 0) != NULL || forge_last_error()[0] == 0) { puts("error path broken"); return 3; }
    printf("consumer_c: libforge %s, abi %u\n", v, (unsigned)forge_abi_version());
    return 0;
}
