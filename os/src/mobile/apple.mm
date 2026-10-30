// forge_mobile iOS / iPadOS backend (Objective-C++, ARC). On macOS this file only provides the null backend.
//   sensors   CMMotionManager (accelerometer, gyro, magnetometer), CMAltimeter (pressure),
//             CLLocationManager (location), UIDevice battery. Light/proximity: not exposed by iOS.
//   touch     FORGETouchView (a UIView subclass the app adds or subclasses) forwards UITouch phases.
//   haptics   UIImpactFeedbackGenerator (amplitude -> intensity); AudioServices vibrate as fallback.
//   display   UIScreen nativeBounds/scale/maximumFramesPerSecond + UIDevice orientation.
//   cameras   AVCaptureDeviceDiscoverySession.  permissions: AVCaptureDevice / CLLocationManager / UNUserNotificationCenter.
#include "internal.hpp"
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_WATCH   // iOS, iPadOS, tvOS-less UIKit targets; watchOS uses the null backend + glue
#import <AVFoundation/AVFoundation.h>
#import <AudioToolbox/AudioToolbox.h>
#import <CoreLocation/CoreLocation.h>
#import <CoreMotion/CoreMotion.h>
#import <LocalAuthentication/LocalAuthentication.h>
#import <UIKit/UIKit.h>
#import <UserNotifications/UserNotifications.h>

using namespace forge::os::mobile;
using forge::os::mobile::detail::queue;

@interface FORGELocationDelegate : NSObject <CLLocationManagerDelegate>
@property (nonatomic) std::shared_ptr<queue> q;
@end
@implementation FORGELocationDelegate
- (void)locationManager:(CLLocationManager*)m didUpdateLocations:(NSArray<CLLocation*>*)locs {
    for (CLLocation* l in locs) _q->push({false, {sensor_kind::location, detail::now_ns(), {l.coordinate.latitude, l.coordinate.longitude, l.altitude, l.horizontalAccuracy}, 4}, {}});
}
@end

// Apps put this view (or a subclass) under their content; every touch reaches forge::os::mobile.
@interface FORGETouchView : UIView
@end
@implementation FORGETouchView
- (instancetype)initWithFrame:(CGRect)f { if ((self = [super initWithFrame:f])) self.multipleTouchEnabled = YES; return self; }
static void forge_send(UIView* v, NSSet<UITouch*>* ts, UIEvent* ev, touch_phase ph) {
    int n = int([ev allTouches].count);
    for (UITouch* t in ts) {
        CGPoint p = [t locationInView:v]; CGFloat s = v.contentScaleFactor;
        double pr = t.maximumPossibleForce > 0 ? t.force / t.maximumPossibleForce : 1.0;
        feed_touch({std::uint64_t(t.timestamp * 1e9), ph, int(reinterpret_cast<std::uintptr_t>((__bridge void*)t) & 0x7fffffff), p.x * s, p.y * s, pr, n});
    }
}
- (void)touchesBegan:(NSSet<UITouch*>*)t withEvent:(UIEvent*)e { forge_send(self, t, e, touch_phase::down); }
- (void)touchesMoved:(NSSet<UITouch*>*)t withEvent:(UIEvent*)e { forge_send(self, t, e, touch_phase::move); }
- (void)touchesEnded:(NSSet<UITouch*>*)t withEvent:(UIEvent*)e { forge_send(self, t, e, touch_phase::up); }
- (void)touchesCancelled:(NSSet<UITouch*>*)t withEvent:(UIEvent*)e { forge_send(self, t, e, touch_phase::cancel); }
@end

namespace forge::os::mobile::detail {
namespace {
template <class F> void on_main(F&& f) { if ([NSThread isMainThread]) f(); else dispatch_sync(dispatch_get_main_queue(), ^{ f(); }); }

// RAII owner of whatever CoreMotion/CoreLocation object feeds one stream; stops updates on destruction.
struct apple_sub {
    CMMotionManager* mm = nil; CMAltimeter* alt = nil; CLLocationManager* lm = nil; FORGELocationDelegate* del = nil;
    NSOperationQueue* oq = nil; sensor_kind k{}; std::shared_ptr<worker> battery;
    ~apple_sub() {
        if (mm) { switch (k) { case sensor_kind::accelerometer: [mm stopAccelerometerUpdates]; break; case sensor_kind::gyroscope: [mm stopGyroUpdates]; break; default: [mm stopMagnetometerUpdates]; } }
        if (alt) [alt stopRelativeAltitudeUpdates];
        if (lm) { CLLocationManager* l = lm; on_main([l] { [l stopUpdatingLocation]; }); }
    }
};

struct apple_backend final : backend {
    CMMotionManager* mm = [CMMotionManager new];
    std::string name() const override { return "ios"; }
    result<std::vector<sensor_info>> sensors() override {
        std::vector<sensor_info> v;
        if (mm.accelerometerAvailable) v.push_back({sensor_kind::accelerometer, "CoreMotion accelerometer", "Apple", 100, 0, 0});
        if (mm.gyroAvailable) v.push_back({sensor_kind::gyroscope, "CoreMotion gyro", "Apple", 100, 0, 0});
        if (mm.magnetometerAvailable) v.push_back({sensor_kind::magnetometer, "CoreMotion magnetometer", "Apple", 100, 0, 0});
        if ([CMAltimeter isRelativeAltitudeAvailable]) v.push_back({sensor_kind::pressure, "CMAltimeter", "Apple", 1, 0, 0});
        if ([CLLocationManager locationServicesEnabled]) v.push_back({sensor_kind::location, "CoreLocation", "Apple", 1, 0, 0});
        v.push_back({sensor_kind::battery, "UIDevice battery", "Apple", 1, 100, 1});
        return {v, {}};
    }
    result<sensor_stream> open_sensor(sensor_kind k, double hz) override {
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q);
        auto sub = std::make_shared<apple_sub>(); sub->k = k; sub->oq = [NSOperationQueue new];
        const double g = 9.80665;
        switch (k) {
        case sensor_kind::accelerometer:
            if (!mm.accelerometerAvailable) return {{}, unsupported("accelerometer")};
            sub->mm = mm; mm.accelerometerUpdateInterval = 1.0 / hz;
            [mm startAccelerometerUpdatesToQueue:sub->oq withHandler:^(CMAccelerometerData* d, NSError*) { if (d) q->push({false, {k, std::uint64_t(d.timestamp * 1e9), {d.acceleration.x * g, d.acceleration.y * g, d.acceleration.z * g}, 3}, {}}); }];
            break;
        case sensor_kind::gyroscope:
            if (!mm.gyroAvailable) return {{}, unsupported("gyroscope")};
            sub->mm = mm; mm.gyroUpdateInterval = 1.0 / hz;
            [mm startGyroUpdatesToQueue:sub->oq withHandler:^(CMGyroData* d, NSError*) { if (d) q->push({false, {k, std::uint64_t(d.timestamp * 1e9), {d.rotationRate.x, d.rotationRate.y, d.rotationRate.z}, 3}, {}}); }];
            break;
        case sensor_kind::magnetometer:
            if (!mm.magnetometerAvailable) return {{}, unsupported("magnetometer")};
            sub->mm = mm; mm.magnetometerUpdateInterval = 1.0 / hz;
            [mm startMagnetometerUpdatesToQueue:sub->oq withHandler:^(CMMagnetometerData* d, NSError*) { if (d) q->push({false, {k, std::uint64_t(d.timestamp * 1e9), {d.magneticField.x, d.magneticField.y, d.magneticField.z}, 3}, {}}); }];
            break;
        case sensor_kind::pressure:
            if (![CMAltimeter isRelativeAltitudeAvailable]) return {{}, unsupported("barometer")};
            sub->alt = [CMAltimeter new];
            [sub->alt startRelativeAltitudeUpdatesToQueue:sub->oq withHandler:^(CMAltitudeData* d, NSError*) { if (d) q->push({false, {k, std::uint64_t(d.timestamp * 1e9), {d.pressure.doubleValue * 10.0 /* kPa -> hPa */}, 1}, {}}); }];
            break;
        case sensor_kind::location: {
            if (![CLLocationManager locationServicesEnabled]) return {{}, unsupported("location services disabled")};
            auto* del = [FORGELocationDelegate new]; del.q = q; sub->del = del;
            __block CLLocationManager* lm = nil;
            on_main([&] { lm = [CLLocationManager new]; lm.delegate = del; [lm requestWhenInUseAuthorization]; [lm startUpdatingLocation]; });
            sub->lm = lm; break;
        }
        case sensor_kind::battery: {
            sub->battery = std::make_shared<worker>(); auto period = std::uint64_t(1e9 / hz);
            on_main([] { [UIDevice currentDevice].batteryMonitoringEnabled = YES; });
            sub->battery->start([q, wp = sub->battery.get(), period] {
                do { UIDevice* d = [UIDevice currentDevice]; if (d.batteryLevel >= 0) q->push({false, {sensor_kind::battery, now_ns(), {d.batteryLevel * 100.0, d.batteryState == UIDeviceBatteryStateCharging || d.batteryState == UIDeviceBatteryStateFull ? 1.0 : 0.0}, 2}, {}}); } while (wp->sleep_ns(period));
            });
            break;
        }
        default: return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " on iOS")};
        }
        return {make_sensor(k, q, sub), {}};
    }
    result<touch_stream> open_touch() override { auto q = std::make_shared<queue>(); bridge::get().add_touch(q); return {make_touch(q, nullptr), {}}; }
    status vibrate(int ms, double amp) override {
        if (auto h = backend::vibrate(ms, amp); !is_unsupported(h.err)) return h;
        on_main([amp] { auto* g = [[UIImpactFeedbackGenerator alloc] initWithStyle:UIImpactFeedbackStyleHeavy]; [g prepare]; [g impactOccurredWithIntensity:amp]; });
        return {};
    }
    result<display_info> display() override {
        if (auto b = backend::display()) return b;
        display_info d;
        on_main([&] {
            UIScreen* s = [UIScreen mainScreen]; d.width_px = int(s.nativeBounds.size.width); d.height_px = int(s.nativeBounds.size.height);
            d.scale = s.nativeScale; d.dpi = 163.0 * s.nativeScale; d.refresh_hz = double(s.maximumFramesPerSecond);
            switch ([UIDevice currentDevice].orientation) {
            case UIDeviceOrientationPortrait: d.orient = orientation::portrait; break;
            case UIDeviceOrientationPortraitUpsideDown: d.orient = orientation::portrait_upside_down; break;
            case UIDeviceOrientationLandscapeLeft: d.orient = orientation::landscape_left; break;
            case UIDeviceOrientationLandscapeRight: d.orient = orientation::landscape_right; break;
            default: d.orient = orientation::unknown;
            }
        });
        return {d, {}};
    }
    result<std::vector<camera_info>> cameras() override {
        std::vector<camera_info> v;
        auto* ds = [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera, AVCaptureDeviceTypeBuiltInUltraWideCamera, AVCaptureDeviceTypeBuiltInTelephotoCamera]
                                                                         mediaType:AVMediaTypeVideo position:AVCaptureDevicePositionUnspecified];
        for (AVCaptureDevice* c in ds.devices) v.push_back({c.uniqueID.UTF8String, c.localizedName.UTF8String, c.position == AVCaptureDevicePositionFront});
        return {v, {}};
    }
    result<permission_state> permission_status(permission p) override {
        if (auto b = backend::permission_status(p)) return b;
        switch (p) {
        case permission::camera: switch ([AVCaptureDevice authorizationStatusForMediaType:AVMediaTypeVideo]) {
            case AVAuthorizationStatusAuthorized: return {permission_state::granted, {}};
            case AVAuthorizationStatusNotDetermined: return {permission_state::not_determined, {}};
            default: return {permission_state::denied, {}}; }
        case permission::location: {
            CLAuthorizationStatus s = [CLLocationManager new].authorizationStatus;
            if (s == kCLAuthorizationStatusNotDetermined) return {permission_state::not_determined, {}};
            if (s == kCLAuthorizationStatusAuthorizedAlways || s == kCLAuthorizationStatusAuthorizedWhenInUse) return {permission_state::granted, {}};
            return {permission_state::denied, {}}; }
        case permission::sensors: switch ([CMMotionActivityManager authorizationStatus]) {
            case CMAuthorizationStatusAuthorized: return {permission_state::granted, {}};
            case CMAuthorizationStatusNotDetermined: return {permission_state::not_determined, {}};
            default: return {permission_state::denied, {}}; }
        case permission::notifications: {
            dispatch_semaphore_t sem = dispatch_semaphore_create(0); __block UNAuthorizationStatus st = UNAuthorizationStatusNotDetermined;
            [[UNUserNotificationCenter currentNotificationCenter] getNotificationSettingsWithCompletionHandler:^(UNNotificationSettings* s) { st = s.authorizationStatus; dispatch_semaphore_signal(sem); }];
            dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC * 2));
            return {st == UNAuthorizationStatusNotDetermined ? permission_state::not_determined : st == UNAuthorizationStatusDenied ? permission_state::denied : permission_state::granted, {}}; }
        case permission::vibrate: return {permission_state::not_required, {}};
        }
        return {permission_state::not_required, {}};
    }
    status request_permission(permission p) override {
        if (auto h = backend::request_permission(p); !is_unsupported(h.err)) return h;
        switch (p) {
        case permission::camera: [AVCaptureDevice requestAccessForMediaType:AVMediaTypeVideo completionHandler:^(BOOL) {}]; return {};
        case permission::location: on_main([] { static CLLocationManager* lm = [CLLocationManager new]; [lm requestWhenInUseAuthorization]; }); return {};
        case permission::notifications:
            [[UNUserNotificationCenter currentNotificationCenter] requestAuthorizationWithOptions:(UNAuthorizationOptionAlert | UNAuthorizationOptionSound) completionHandler:^(BOOL, NSError*) {}];
            return {};
        default: return {};
        }
    }
    status notify(std::string_view t, std::string_view b) override {
        if (auto h = backend::notify(t, b); !is_unsupported(h.err)) return h;
        auto* c = [UNMutableNotificationContent new];
        c.title = [[NSString alloc] initWithBytes:t.data() length:t.size() encoding:NSUTF8StringEncoding];
        c.body = [[NSString alloc] initWithBytes:b.data() length:b.size() encoding:NSUTF8StringEncoding];
        auto* r = [UNNotificationRequest requestWithIdentifier:[NSUUID UUID].UUIDString content:c trigger:nil];
        [[UNUserNotificationCenter currentNotificationCenter] addNotificationRequest:r withCompletionHandler:nil];
        return {};
    }
    static NSString* ns(std::string_view s) { return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding]; }
    result<double> brightness() override { double v = 0; on_main([&] { v = UIScreen.mainScreen.brightness; }); return {v, {}}; }
    status set_brightness(double l) override { on_main([l] { UIScreen.mainScreen.brightness = CGFloat(l); }); return {}; }
    status keep_awake(bool on) override { on_main([on] { UIApplication.sharedApplication.idleTimerDisabled = on; }); return {}; }
    status torch(bool on) override {
        AVCaptureDevice* d = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
        if (!d || !d.hasTorch) return {{}, unsupported("torch (no flash)")};
        NSError* e = nil; if (![d lockForConfiguration:&e]) return {{}, {1, "torch: lockForConfiguration failed"}};
        d.torchMode = on ? AVCaptureTorchModeOn : AVCaptureTorchModeOff; [d unlockForConfiguration]; return {};
    }
    status open_url(std::string_view u) override {
        NSURL* url = [NSURL URLWithString:ns(u)]; if (!url) return {{}, {22, "open_url: bad URL"}};
        on_main([url] { [UIApplication.sharedApplication openURL:url options:@{} completionHandler:nil]; }); return {};
    }
    result<biometric_info> biometrics() override {
        LAContext* c = [LAContext new]; NSError* e = nil;
        bool ok = [c canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&e];
        biometric_info b; b.available = c.biometryType != LABiometryTypeNone; b.enrolled = ok;
        b.kind = c.biometryType == LABiometryTypeFaceID ? "face" : c.biometryType == LABiometryTypeTouchID ? "fingerprint" : "";
        return {b, {}};
    }
    result<device_info> info() override {
        auto d = backend::info(); if (!d) return d;
        UIDevice* u = UIDevice.currentDevice; d.value.manufacturer = "Apple"; d.value.model = u.model.UTF8String;
        d.value.os_name = u.systemName.UTF8String; d.value.os_version = u.systemVersion.UTF8String;
        d.value.form = u.userInterfaceIdiom == UIUserInterfaceIdiomPad ? form_factor::tablet : u.userInterfaceIdiom == UIUserInterfaceIdiomTV ? form_factor::tv
                     : u.userInterfaceIdiom == UIUserInterfaceIdiomCarPlay ? form_factor::car : form_factor::phone;
        return d;
    }
    void lifecycle_started() override {
        backend::lifecycle_started();
        static std::once_flag once; std::call_once(once, [] {
            auto* nc = NSNotificationCenter.defaultCenter;
            [nc addObserverForName:UIApplicationDidBecomeActiveNotification object:nil queue:nil usingBlock:^(NSNotification*) { dispatch_lifecycle(app_event::foreground); }];
            [nc addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil queue:nil usingBlock:^(NSNotification*) { dispatch_lifecycle(app_event::background); }];
            [nc addObserverForName:UIApplicationDidReceiveMemoryWarningNotification object:nil queue:nil usingBlock:^(NSNotification*) { dispatch_lifecycle(app_event::low_memory); }];
            [nc addObserverForName:UIApplicationWillTerminateNotification object:nil queue:nil usingBlock:^(NSNotification*) { dispatch_lifecycle(app_event::terminate); }];
        });
    }
};
} // namespace
std::shared_ptr<backend> make_apple() { return std::make_shared<apple_backend>(); }
} // namespace forge::os::mobile::detail
#elif defined(__APPLE__) && TARGET_OS_OSX
// macOS: trackpad touches/gestures come from the app's NSView via feed_touch; location via CoreLocation in
// the app (feed_sensor). Here: IOKit power assertions (keep-awake), NSWorkspace open_url, user notifications,
// device info; everything else uses the portable defaults (power, clipboard, connectivity) or is not_supported.
#import <AppKit/AppKit.h>
#import <IOKit/pwr_mgt/IOPMLib.h>
#import <UserNotifications/UserNotifications.h>
namespace forge::os::mobile::detail {
namespace {
struct mac_backend final : backend {
    IOPMAssertionID awake = 0;
    ~mac_backend() override { if (awake) IOPMAssertionRelease(awake); }
    std::string name() const override { return "macos"; }
    result<touch_stream> open_touch() override { auto q = std::make_shared<queue>(); bridge::get().add_touch(q); return {make_touch(q, nullptr), {}}; }
    result<sensor_stream> open_sensor(sensor_kind k, double) override {
        if (k != sensor_kind::location) return {{}, unsupported(std::string("sensor ") + sensor_name(k) + " on macOS")};
        auto q = std::make_shared<queue>(); bridge::get().add_sensor(k, q); return {make_sensor(k, q, nullptr), {}};
    }
    status keep_awake(bool on) override {
        if (on && !awake) return IOPMAssertionCreateWithName(kIOPMAssertionTypeNoDisplaySleep, kIOPMAssertionLevelOn, CFSTR("forge keep_awake"), &awake) == kIOReturnSuccess ? status{} : status{{}, {1, "IOPMAssertionCreateWithName failed"}};
        if (!on && awake) { IOPMAssertionRelease(awake); awake = 0; }
        return {};
    }
    status open_url(std::string_view u) override {
        NSURL* url = [NSURL URLWithString:[[NSString alloc] initWithBytes:u.data() length:u.size() encoding:NSUTF8StringEncoding]];
        if (!url) return {{}, {22, "open_url: bad URL"}};
        return [NSWorkspace.sharedWorkspace openURL:url] ? status{} : status{{}, {1, "NSWorkspace openURL failed"}};
    }
    result<device_info> info() override {
        auto d = backend::info(); if (!d) return d; d.value.manufacturer = "Apple";
        d.value.form = power_status().has_battery ? form_factor::laptop : form_factor::desktop; return d;
    }
    result<permission_state> permission_status(permission p) override { if (auto b = backend::permission_status(p)) return b; return {permission_state::not_required, {}}; }
    status request_permission(permission) override { return {}; }
};
} // namespace
std::shared_ptr<backend> make_apple() { return std::make_shared<mac_backend>(); }
} // namespace forge::os::mobile::detail
#elif defined(__APPLE__)
namespace forge::os::mobile::detail { std::shared_ptr<backend> make_apple() { return make_null(); } }   // watchOS: glue only
#endif
