
use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;
use raw_window_handle::HasRawDisplayHandle;

pub fn required_extension_names(window:&winit::window::Window, debug:bool) -> Vec<*const i8> {
    let extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle()).unwrap();
    let mut v = extension_names.to_vec();
    if debug{
        v.push( DebugUtils::name().as_ptr());
        v.push(b"VK_EXT_validation_features\0".as_ptr() as *const i8);
    }
    v
}
