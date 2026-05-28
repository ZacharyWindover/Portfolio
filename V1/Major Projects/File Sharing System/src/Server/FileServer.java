package Server;

import java.io.*;

import java.net.*;


public class FileServer {

    protected   static        Socket                clientSocket =  null;
    protected   static        ServerSocket          serverSocket =  null;
    protected   static        FileServerThread[]    threads      =  null;

    public      static        int                   SERVER_PORT  =  8001;
    public      static  final int                   MAX_CLIENTS  =  100;
    protected   static        int                   numClients   =  0;

    protected   static      String                  server_IP;

    public FileServer() {

        try {
            InetAddress iAddress = InetAddress.getLocalHost();
            server_IP = iAddress.getHostAddress();
            System.out.println("Server IP Address: " + server_IP);
        } catch (UnknownHostException e) {
            System.out.println("L");
        }


        try {
            serverSocket = new ServerSocket(SERVER_PORT);
            threads = new FileServerThread[MAX_CLIENTS];
            System.out.println("SERVER_PORT: " + SERVER_PORT);

            while(true) {
                clientSocket = serverSocket.accept();
                threads[numClients] = new FileServerThread(clientSocket);
                threads[numClients].start();
                numClients++;
            }

        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("IOException while creating server connection.");
        }

    }

    public static void main(String[] args) {

        FileServer application = new FileServer();

    }

}
