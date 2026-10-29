#pragma once
// BSD sockets on every platform (Winsock2 on Windows). Call forge::net_init() before the first socket.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using forge_sock_t = SOCKET;
#define FORGE_SOCK_BAD INVALID_SOCKET
#define forge_sock_close closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using forge_sock_t = int;
#define FORGE_SOCK_BAD (-1)
#define forge_sock_close ::close
#endif
#if defined(_WIN32)
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
using socklen_t = int;
#endif
namespace forge {
inline void net_init() {
#if defined(_WIN32)
    static bool ok = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d) == 0; }(); (void)ok;
#endif
}
} // namespace forge
