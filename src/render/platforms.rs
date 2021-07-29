use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::XlibSurface;
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;

fn append_debug(mut v:Vec<*const i8>,debug:bool)->Vec<*const i8>{
    if debug{
        v.push( DebugUtils::name().as_ptr());
        v.push(b"VK_EXT_validation_features\0".as_ptr() as *const i8);
        v
    }else{
        v
    }
}

// required extension ------------------------------------------------------
#[cfg(target_os = "macos")]
pub fn required_extension_names(debug:bool) -> Vec<*const i8> {
    append_debug(vec![
        Surface::name().as_ptr(),
        MacOSSurface::name().as_ptr(),
        b"VK_KHR_get_physical_device_properties2\0".as_ptr() as *const i8
    ],debug)
}

#[cfg(all(windows))]
pub fn required_extension_names(debug:bool) -> Vec<*const i8> {
    append_debug(vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr()
    ],debug)
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names(debug:bool) -> Vec<*const i8> {
    append_debug(vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
    ],debug)
}