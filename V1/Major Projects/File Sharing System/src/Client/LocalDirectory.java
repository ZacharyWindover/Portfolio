package Client;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileOwnerAttributeView;
import java.text.SimpleDateFormat;

public class LocalDirectory {

    protected static File folder;
    protected static File[] files;
    protected static String[] fileNames;
    protected static String[] fileSizes;
    protected static String[] lastUpdated;
    protected static String[] fileAuthor;

    LocalDirectory() throws IOException {
        this.folder = new File("./src/Client/resources/local/");
        this.files = folder.listFiles();

        this.fileNames = new String[files.length];
        this.fileSizes = new String[files.length];
        this.lastUpdated = new String[files.length];
        this.fileAuthor = new String[files.length];

        setFileNames();
        setFileSizes();
        setLastUpdated();
        setFileAuthor();

    }

    public static String[] getFileNames() { return fileNames; }
    public static String[] getFileSizes() { return fileSizes; }
    public static String[] getLastUpdated() { return lastUpdated; }
    public static String[] getFileAuthor() { return fileAuthor; }

    private static void setFileNames() {
        for (int x = 0; x < files.length; x++) {
            fileNames[x] = files[x].getName();
            //System.out.println("fileNames[x] = " + fileNames[x]);
        }
    }

    private static void setFileSizes() {
        for (int x = 0; x < files.length; x++) {
            long bytes = files[x].length();
            int mb = (int) Math.floor(bytes / 1000);
            fileSizes[x] = (mb + " KB");
        }
    }

    private static void setLastUpdated() {
        for (int x = 0; x < files.length; x++) {
            SimpleDateFormat sdf = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
            lastUpdated[x] = sdf.format(files[x].lastModified());
        }

    }

    private static void setFileAuthor() throws IOException {
        for (int x = 0; x < files.length; x++) {
            Path path = files[x].getAbsoluteFile().toPath();
            FileOwnerAttributeView info = Files.getFileAttributeView(path, FileOwnerAttributeView.class);
            fileAuthor[x] = String.valueOf(info.getOwner());
        }
    }

}
