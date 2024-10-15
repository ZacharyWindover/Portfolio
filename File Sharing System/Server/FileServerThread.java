package Server;

import java.io.*;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;

public class FileServerThread extends Thread {

    protected static Socket                 socket      =   null;

    protected static PrintWriter            networkOut  =   null;
    protected static BufferedReader         in          =   null;

    FileServerThread(Socket socket) {
        super();
        this.socket = socket;

        System.out.println("Socket: " + socket);

        try {
            networkOut = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
        }
        catch (IOException e) { System.err.println("IOException while opening a read/write connection"); }

    }

    public void run() {

        boolean endOfSession = false;

        while (!endOfSession) {
            try {
                endOfSession = processCommand();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            socket.close();
        } catch (IOException  e) {
            e.printStackTrace();
        }
    }

    protected boolean processCommand() throws IOException {
        String input = null;

        try {
            input = in.readLine();

        } catch (IOException e) {
            System.err.println("Error reading command from socket.");
        }

        if (input == null) {

        } else {

            if (input.equalsIgnoreCase(("DIR"))) { processCommand(input, null); }
            else {
                String[] parts = input.split(" ", 2);
                String command = parts[0];
                String args = parts[1];

                if (input.contains("UPLOAD")) { processCommand(command, args); }
                else if (input.contains("DOWNLOAD")) { processCommand(command, args); }
                else { System.err.println("ERROR PASSING COMMAND TO SERVER / SERVER ERROR READING COMMAND"); }

            }

        }

        return true;

    }

    protected void processCommand(String command, String args) {

        if (command.equalsIgnoreCase("DIR")) { sendDirectory(); }

        else if (command.equalsIgnoreCase("UPLOAD")) { uploadFile(args); }

        else if (command.equalsIgnoreCase(("DOWNLOAD"))) { downloadFile(args); }

        else { System.err.println("ERROR 404, COMMAND NOT RECOGNIZED"); }


    }

    protected void sendDirectory() {

        DataOutputStream DOS = null;

        try { DOS = new DataOutputStream(socket.getOutputStream()); }
        catch (IOException e) {
            System.err.println("Failed to create DataOutputStream");
            e.printStackTrace();
        }

        //System.out.println("Sending shared directory");

        // get list of files in directory and send to client
        SharedDirectory sharedDirectory = null;

        try { sharedDirectory = new SharedDirectory(); }
        catch (IOException e) { e.printStackTrace(); }

        String[] fileNames = sharedDirectory.getFileNames();
        String[] fileSizes = sharedDirectory.getFileSizes();
        String[] lastUpdates = sharedDirectory.getLastUpdated();
        String[] fileAuthors = sharedDirectory.getFileAuthor();

        try { DOS.writeBytes(String.valueOf(fileNames.length + "\n")); }
        catch (IOException e) { e.printStackTrace(); }

        for (int x = 0; x < fileNames.length; x++) {
            try { DOS.writeBytes(fileNames[x] + "\n"); }
            catch (IOException e) { e.printStackTrace(); }
        }

        for (int x = 0; x < fileNames.length; x++) {
            try { DOS.writeBytes(fileSizes[x] + "\n"); }
            catch (IOException e) { e.printStackTrace(); }
        }

        for (int x = 0; x < fileNames.length; x++) {
            try { DOS.writeBytes(lastUpdates[x] + "\n"); }
            catch (IOException e) { e.printStackTrace(); }
        }

        for (int x = 0; x < fileNames.length; x++) {
            try { DOS.writeBytes(fileAuthors[x] + "\n"); }
            catch (IOException e) { e.printStackTrace(); }
        }

        // close DOS
        try {
            DOS.flush();
            DOS.close();
        }
        catch (IOException e) { System.err.println("Failed to flush and close DOS"); }


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
    protected void uploadFile(String args) {

        int selectedFileSize = 0;

        String fileName = args;
        String filePath = "./src/Server/resources/shared/" + fileName;

        File output = new File(filePath);

        // create DataInputStream
        DataInputStream DIS = null;
        try { DIS = new DataInputStream(new BufferedInputStream(socket.getInputStream())); }
        catch (IOException e) { System.err.println("Failed in creating data input stream"); }

        // get file size from client
        try { selectedFileSize = DIS.readInt(); }
        catch (IOException e) { System.err.println("Failed to retrieve file size from client"); }

        // getting file from client
        byte[] byteArray = new byte[selectedFileSize];
        try { DIS.readFully(byteArray, 0, byteArray.length); }
        catch (IOException e) { System.err.println("Failed to get file from client"); }

        try { DIS.close(); }
        catch (IOException e) { System.err.println("Failed to close DataInputStream for Server Upload File"); }


        int copyIndex = 1;
        String indexed = " (" + String.valueOf(copyIndex) + ")";

        try {

            // if output already exists, give it a different name by adding index to end
            if (output.exists()) {

                while (output.exists()) {
                    filePath = "./src/Server/resources/shared/" + fileName;
                    filePath = filePath.substring(0, filePath.length() - 4) + indexed + ".pdf";
                    output = new File(filePath);
                    copyIndex++;
                    indexed = " (" + String.valueOf(copyIndex) + ")";
                }
            }

            // create new file with output
            output.createNewFile();

        } catch (IOException e) { System.err.println("Failed to create new file"); }

        OutputStream fileOutput = null;

        try { fileOutput = new FileOutputStream(output); }
        catch (FileNotFoundException e) { System.err.println("Failed to create output stream"); }

        // writing to file
        try { fileOutput.write(byteArray); }
        catch (IOException e) { System.err.println("Failed to write byteArray to fileOutput"); }

        // closing stream
        try { fileOutput.close(); }
        catch (IOException e) { System.err.println("Failed to close fileOutput"); }

    }

    /*
    * Order of operations:
    * - set requested file name from args
    * - set the requested file's path
    * - create an output file with the file path
    * - create byteArray
    * - get all bytes from the file and record in byteArray
    * - create DataOutputStream
    * - send byteArray to client
    * - close DOS
     */
    protected void downloadFile(String args) {

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

        // getting file name, path, size and author
        String requestedFileName = args;
        String requestedFilePath = ("./src/Server/resources/shared/" + args);

        // creating path
        Path filePath = Path.of(requestedFilePath);

        // creating byteArray
        byte[] byteArray = null;
        try { byteArray = Files.readAllBytes(filePath); }
        catch (IOException e) {
            System.err.println("args passed: " + args);
            System.err.println("requestedFilePath: " + requestedFilePath);
            System.err.println("Failed to write file bytes to byteArray");
            e.printStackTrace();
        }

        // creating DOS
        DataOutputStream DOS = null;
        try { DOS = new DataOutputStream(socket.getOutputStream()); }
        catch (IOException e) { System.err.println("Failed to create DOS for DOWNLOAD " + args); }

        // sending file size to client
        try { DOS.writeInt(byteArray.length); }
        catch (IOException e) { System.err.println("Failed to send byteArray size to client"); }

        // sending bytes to client
        try { DOS.write(byteArray, 0, byteArray.length); }
        catch (IOException e) {
            System.err.println("Failed to send byteArray to client");
            System.err.println("args: " + args);
            System.err.println("requestedFilePath: " + requestedFilePath);
            e.printStackTrace();
        }

        // closing DOS
        try {
            DOS.flush();
            DOS.close();
        } catch (IOException e) { System.err.println("Failed to flush and close DOS for DOWNLOAD " + args); }


    }




}
