/**
 * Native File Dialog Implementation
 * Cross-platform file picker using system dialogs
 */

#include <string>
#include <vector>
#include <cstdlib>
#include <array>
#include <memory>

#ifdef _WIN32
    #include <windows.h>
    #include <shobjidl.h>
#elif __APPLE__
    #include <CoreFoundation/CoreFoundation.h>
    #include <Cocoa/Cocoa.h>
#else // Linux
    #include <cstdio>
#endif

namespace vk_symbiote {
namespace gui {

class NativeFileDialog {
public:
    // Open file dialog - returns selected file path or empty string
    static std::string openFile(const std::vector<std::pair<std::string, std::string>>& filters = {}) {
#ifdef _WIN32
        return openFileWindows(filters);
#elif __APPLE__
        return openFileMacOS(filters);
#else
        return openFileLinux(filters);
#endif
    }
    
    // Open folder dialog
    static std::string openFolder() {
#ifdef _WIN32
        return openFolderWindows();
#elif __APPLE__
        return openFolderMacOS();
#else
        return openFolderLinux();
#endif
    }
    
    // Save file dialog
    static std::string saveFile(const std::string& default_name, 
                                 const std::vector<std::pair<std::string, std::string>>& filters = {}) {
#ifdef _WIN32
        return saveFileWindows(default_name, filters);
#elif __APPLE__
        return saveFileMacOS(default_name, filters);
#else
        return saveFileLinux(default_name, filters);
#endif
    }

private:
#ifdef _WIN32
    static std::string openFileWindows(const std::vector<std::pair<std::string, std::string>>& filters) {
        // Windows IFileDialog implementation
        // Placeholder for actual implementation
        return "";
    }
    
    static std::string openFolderWindows() {
        // Windows folder picker
        return "";
    }
    
    static std::string saveFileWindows(const std::string& default_name,
                                       const std::vector<std::pair<std::string, std::string>>& filters) {
        return "";
    }
    
#elif __APPLE__
    static std::string openFileMacOS(const std::vector<std::pair<std::string, std::string>>& filters) {
        // macOS NSOpenPanel implementation
        return "";
    }
    
    static std::string openFolderMacOS() {
        return "";
    }
    
    static std::string saveFileMacOS(const std::string& default_name,
                                     const std::vector<std::pair<std::string, std::string>>& filters) {
        return "";
    }
    
#else // Linux - use zenity or kdialog
    // Custom deleter to avoid GCC warning about function attributes
    struct PipeDeleter {
        void operator()(FILE* pipe) const {
            if (pipe) pclose(pipe);
        }
    };
    
    static std::string execCommand(const char* cmd) {
        std::array<char, 128> buffer;
        std::string result;
        std::unique_ptr<FILE, PipeDeleter> pipe(popen(cmd, "r"));
        if (!pipe) return "";
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        // Remove trailing newline
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
        return result;
    }
    
    static std::string openFileLinux(const std::vector<std::pair<std::string, std::string>>& filters) {
        // Try zenity first, then kdialog
        std::string cmd = "zenity --file-selection --title='Select Model File' ";
        
        // Add filters
        if (!filters.empty()) {
            cmd += "--file-filter='Model Files ";
            for (const auto& filter : filters) {
                cmd += "*." + filter.first + " ";
            }
            cmd += "' ";
        }
        
        std::string result = execCommand(cmd.c_str());
        return result;
    }
    
    static std::string openFolderLinux() {
        std::string cmd = "zenity --file-selection --directory --title='Select Project Folder'";
        return execCommand(cmd.c_str());
    }
    
    static std::string saveFileLinux(const std::string& default_name,
                                     const std::vector<std::pair<std::string, std::string>>& filters) {
        std::string cmd = "zenity --file-selection --save --confirm-overwrite ";
        cmd += "--filename='" + default_name + "' ";
        return execCommand(cmd.c_str());
    }
#endif
};

} // namespace gui
} // namespace vk_symbiote
