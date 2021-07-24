pub const APP_TITLE: &'static str = "Evolut.io";
pub const APP_INFO: ash::vk::ApplicationInfo = ash::vk::ApplicationInfo {
    s_type: ash::vk::StructureType::APPLICATION_INFO,
    p_next: std::ptr::null(),
    p_application_name: b"Evolut.io\0".as_ptr() as *const i8,
    application_version: 0,
    p_engine_name: b"Vulkan\0".as_ptr() as *const i8,
    engine_version: 0,
    api_version: 0,
};