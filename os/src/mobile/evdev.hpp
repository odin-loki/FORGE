#pragma once
// evdev multitouch reader shared by the Linux and FreeBSD backends (protocol B slots + single-touch BTN_TOUCH).
#include "internal.hpp"
#include <algorithm>
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#if defined(__FreeBSD__)
#include <dev/evdev/input.h>
#ifndef input_event_sec
#define input_event_sec time.tv_sec
#define input_event_usec time.tv_usec
#endif
#else
#include <linux/input.h>
#endif
namespace forge::os::mobile::detail {
inline void evdev_loop(int fd, queue& q, worker& w) {
    struct slot { int id = -1; double x = 0, y = 0, p = 1; bool changed = false, down = false, up = false; };
    std::vector<slot> slots(16); int cur = 0; bool st_touch = false, st_changed = false;
    auto active = [&] { int n = 0; for (auto& s : slots) n += s.id >= 0; return n; };
    input_event buf[64];
    while (!w.stop) {
        pollfd pf{fd, POLLIN, 0}; int r = ::poll(&pf, 1, 50);
        if (r < 0 && errno != EINTR) break; if (r <= 0) continue;
        ssize_t n = ::read(fd, buf, sizeof buf);
        if (n == 0) break; if (n < 0) { if (errno == EAGAIN || errno == EINTR) continue; break; }
        for (std::size_t i = 0; i < std::size_t(n) / sizeof(input_event); ++i) {
            auto& ev = buf[i]; auto& s = slots[std::size_t(cur)];
            if (ev.type == EV_ABS) {
                switch (ev.code) {
                case ABS_MT_SLOT: cur = std::clamp(ev.value, 0, 15); break;
                case ABS_MT_TRACKING_ID: if (ev.value >= 0) { s.id = ev.value; s.down = true; } else { s.up = true; } s.changed = true; break;
                case ABS_MT_POSITION_X: s.x = ev.value; s.changed = true; break;
                case ABS_MT_POSITION_Y: s.y = ev.value; s.changed = true; break;
                case ABS_MT_PRESSURE: s.p = ev.value; s.changed = true; break;
                case ABS_X: if (s.id < 0 || cur == 0) { slots[0].x = ev.value; st_changed = true; } break;
                case ABS_Y: if (s.id < 0 || cur == 0) { slots[0].y = ev.value; st_changed = true; } break;
                }
            } else if (ev.type == EV_KEY && ev.code == BTN_TOUCH) {
                st_touch = true; if (ev.value) { if (slots[0].id < 0) { slots[0].id = 0; slots[0].down = true; } } else slots[0].up = true;
                slots[0].changed = true;
            } else if (ev.type == EV_SYN && ev.code == SYN_REPORT) {
                std::uint64_t t = std::uint64_t(ev.input_event_sec) * 1000000000ull + std::uint64_t(ev.input_event_usec) * 1000ull;
                if (st_touch && st_changed) slots[0].changed = true;
                for (auto& sl : slots) {
                    if (!sl.changed) continue;
                    touch_phase ph = sl.up ? touch_phase::up : sl.down ? touch_phase::down : touch_phase::move;
                    int id = sl.id; int ptrs = active();
                    q.push({true, {}, {t, ph, id, sl.x, sl.y, sl.p, ptrs}});
                    if (sl.up) sl.id = -1;
                    sl.changed = sl.down = sl.up = false;
                }
                st_changed = false;
            }
        }
    }
    q.finish();
}

} // namespace forge::os::mobile::detail
