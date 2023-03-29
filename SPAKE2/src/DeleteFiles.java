import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.concurrent.TimeUnit;

public class DeleteFiles {

    public static void main(String[] args) {
        String path = "C:\\Users\\LEGION\\Desktop\\SPAKE2";
        String extension = ".txt";
        deleteFile(path, extension);

    }

    public static void deleteFile(String Path, String extension) {

        File directory = new File(Path);
        File[] files = directory.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.getName().endsWith(extension)) {
                    if (file.delete()) {
                        System.out.println("File deleted successfully: " + file.getAbsolutePath());
                    } else {
                        System.out.println("Failed to delete file: " + file.getAbsolutePath());
                    }
                }
            }
        }
    }
}



