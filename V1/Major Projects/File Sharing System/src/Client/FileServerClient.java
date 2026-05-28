package Client;

import java.io.*;

import java.net.*;

import java.nio.file.Files;
import java.nio.file.Path;

public class FileServerClient {

    private     static      String server_IP = "10.0.0.73";
    private     static      String client_IP;

    private     static      Socket                  socket          =   null;

    private     static      BufferedReader          in              =   null;
    private     static      PrintWriter             networkOut      =   null;

    //public      static      String                  SERVER_ADDRESS  =   "localhost";
    public      static      String                  SERVER_ADDRESS  =   "10.0.0.50";
    public      static      int                     SERVER_PORT     =   8000;

    public      static      String                  command         =   null;

    private     static      String[]                fileNames       =   null;
    private     static      String[]                fileSizes       =   null;
    private     static      String[]                lastUpdates     =   null;
    private     static      String[]                fileAuthors     =   null;

    public FileServerClient(String cmd) throws IOException {

        try {
            InetAddress iAddress = InetAddress.getLocalHost();
            client_IP = iAddress.getHostAddress();
            System.out.println("Client IP Address: " + client_IP);
        } catch (UnknownHostException e) {
            System.out.println("Failed to get client IP");
        }

        //socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
        socket = new Socket(server_IP, SERVER_PORT);

        System.out.println("socket: " + socket);

        if (socket == null) { System.err.println("Socket is null"); }

        try {
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            networkOut = new PrintWriter(socket.getOutputStream(), true);
        }
        catch (IOException e) { System.err.println("IOException while opening a read/write connection"); }

        command = cmd;

        if (command.equalsIgnoreCase("DIR")) {
            setDirectory();
        } else if (command.contains("UPLOAD")) {
            sendFile();
        } else if (command.contains("DOWNLOAD")) {
            receiveFile();
        } else { System.err.println("VALID COMMAND NOT DETECTED"); }

    }

    public void setDirectory() {

        networkOut.println(command);
        int numFiles = 0;

        try { numFiles = Integer.parseInt(in.readLine()); }
        catch (IOException e) { System.err.println("Failed to read number of files from server"); }

        // getting / setting fileNames
        String[] fileNames = new String[numFiles];
        try { for (int x = 0; x < numFiles; x++) { fileNames[x] = in.readLine(); } }
        catch (IOException a) { System.err.println("Failed to get file names from server"); }
        this.fileNames = fileNames;

        // getting / setting fileSizes
        String[] fileSizes = new String[numFiles];
        try { for (int x = 0; x < numFiles; x++) { fileSizes[x] = in.readLine(); } }
        catch (IOException b) { System.err.println("Failed to get file sizes from server"); }
        this.fileSizes = fileSizes;

        // getting / setting lastUpdates
        String[] lastUpdates = new String[numFiles];
        try { for (int x = 0; x < numFiles; x++) { lastUpdates[x] = in.readLine(); } }
        catch (IOException c) { System.err.println("Failed to get last updated times from server"); }
        this.lastUpdates = lastUpdates;

        // getting / setting fileAuthors
        String[] fileAuthors = new String[numFiles];
        try { for (int x = 0; x < numFiles; x++) { fileAuthors[x] = in.readLine(); } }
        catch (IOException d) { System.err.println("Failed to get file authors from server"); }
        this.fileAuthors = fileAuthors;

    }

    /*
     * Order of operations:
     * - set requested file name from args
     * - set the requested file's path
     * - create an output file with the file path
     * - create a DataInputStream to get the bytes from the server
     * - get file size from client
     * - get all bytes from the client
     * - close the DIS
     * - if the file name already exists, add an index number to the end of it, and create the new file
     * - create a new OutputStream to write the bytes to the file;
     * - write bytes to the file
     * - close OutputStream
     */
    public void sendFile() {

        // getting file name, path, size and author
        String requestedFileName = GUI.selectedFileName;
        String requestedFilePath = ("./src/Client/resources/local/" + GUI.selectedFileName);
        System.out.println("requestedFilePath: " + requestedFilePath);

        // send submission ping (contains file name)
        networkOut.println(command);

        Path filePath = Path.of(requestedFilePath);

        byte[] byteArray = null;
        DataOutputStream DOS = null;

        // reading file bytes to byte array
        try { byteArray = Files.readAllBytes(filePath); }
        catch (IOException e) { System.err.println("Failed reading bytes from PDF to byte array"); }

        // creating data output stream
        try { DOS = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream())); }
        catch (IOException e) { System.err.println("Failed in creating Data Output Stream."); }

        // set file length
        int fileLength = (int)byteArray.length;

        // send file size to server
        try { DOS.writeInt(fileLength); }
        catch (IOException e) { System.err.println("Failed in sending the file length to the server."); }

        // send byteArray to server
        try { DOS.write(byteArray); }
        catch (IOException e) {
            System.err.println("Failed in sending byteArray to server.");
            for (int x = 0; x < byteArray.length; x++) { System.out.println(byteArray[x]); }
            e.printStackTrace();
        }

        // flushing and closing DOS
        try { DOS.flush(); DOS.close(); }
        catch (IOException e) { System.err.println("Failed in flushing or closing DOS"); }

    }

    /*
     * Order of operations:
     * - send download ping to server
     * - set the requested file's name, path, size, and author
     * - create an output file with the file path
     * - create a DataInputStream to get the bytes from the server
     * - get all bytes from the server
     * - close the DIS
     * - if the file name already exists, add an index number to the end of it, and create the new file
     * - create a new OutputStream to write the bytes to the file;
     * - write bytes to the file
     * - close OutputStream
     */
    public void receiveFile() {

        /*
        String testFilePath = "./src/Client/resources/local/fifth-business.pdf";
        File testFile = new File(testFilePath);

        try {
            byte[] testByte = Files.readAllBytes(testFile.toPath());

            System.out.println("length of testByte: " + testByte.length);
            System.out.println("testByte.toString(): " + testByte.toString());
            File testFile2 = new File("./src/Client/resources/local/fifth-business2.pdf");
            OutputStream oot = new FileOutputStream(testFile2);
            oot.write(testByte);
            oot.close();

        } catch (IOException e) {
            System.out.println("Test failed");
        }
         */

        networkOut.println(command);

        String requestedFileName = GUI.selectedFileName;
        String requestedFilePath = ("./src/Client/resources/local/" + GUI.selectedFileName);
        String requestedFileSize = GUI.selectedFileSize;
        String requestedFileAuthor = GUI.selectedFileAuthor;

        String[] parts = requestedFileSize.split(" ");
        int selectedFileSize = Integer.parseInt(parts[0]) * 1000;

        File output = new File(requestedFilePath);

        // creating DIS
        DataInputStream DIS = null;
        try { DIS = new DataInputStream(socket.getInputStream()); }
        catch (IOException e) { System.out.println("Failed to create input stream"); }

        // get file size from server
        try { selectedFileSize = DIS.readInt(); }
        catch (IOException e) { System.err.println("Failed to get file size from client"); }

        byte[] byteArray = new byte[selectedFileSize];

        try { DIS.readFully(byteArray, 0, byteArray.length); }
        catch (IOException e) {
            System.out.println("Failed to read file into byte array");
            e.printStackTrace();
        }

        // close the data input stream
        try { DIS.close(); }
        catch (IOException e) { System.err.println("Failed to close DataInputStream DIS"); }

        int copyIndex = 1;
        String indexed = " (" + String.valueOf(copyIndex) + ")";

        try {
            // if output already exists, give it a different name by adding index to end
            if (output.exists()) {

                while (output.exists()) {
                    requestedFilePath = ("./src/Client/resources/local/" + GUI.selectedFileName);
                    requestedFilePath = requestedFilePath.substring(0, requestedFilePath.length() - 4) + indexed + ".pdf";
                    output = new File(requestedFilePath);
                    copyIndex++;
                    indexed = " (" + String.valueOf(copyIndex) + ")";
                }
            }

            // create new file with output
            output.createNewFile();

        } catch (IOException e) { System.err.println("Failed to create new output file"); }

        // create file output stream
        OutputStream fileOutput = null;
        try { fileOutput = new FileOutputStream(output); }
        catch (FileNotFoundException e) { System.err.println("Failed to create fileOutput stream"); }

        // writing to file
        try { fileOutput.write(byteArray); }
        catch (IOException e) { System.err.println("Failed to write byteArray to fileOutput"); }

        // closing stream
        try { fileOutput.close(); }
        catch (IOException e) { System.err.println("Failed to close fileOutput"); }

    }

    public static String[] getFileNames() { return fileNames; }
    public static String[] getFileSizes() { return fileSizes; }
    public static String[] getLastUpdates() { return lastUpdates; }
    public static String[] getFileAuthors() { return fileAuthors; }
}