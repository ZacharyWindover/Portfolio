package Client;

import javafx.application.Application;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;

import javafx.event.ActionEvent;
import javafx.event.EventHandler;

import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.GridPane;

import javafx.stage.Stage;

import java.io.IOException;

public class GUI extends Application {

    public static String command = null;

    public static String[] sharedFileNames;
    public static String[] sharedFileSizes;
    public static String[] sharedLastUpdates;
    public static String[] sharedFileAuthors;

    public static String[] localFileNames;
    public static String[] localFileSizes;
    public static String[] localLastUpdates;
    public static String[] localFileAuthors;

    public static ListView sharedListView;

    public static ListView localListView;

    public static String selectedFileName;
    public static String selectedFileAuthor;
    public static String selectedFileSize;



    @Override
    public void start(Stage primaryStage) throws Exception {
        BorderPane bp = borderPaneSetup();

        primaryStage.setTitle("File Sharing Server Client");
        primaryStage.setScene(new Scene(bp, 1280, 720));
        primaryStage.show();

    }

    public BorderPane borderPaneSetup() throws IOException {
        BorderPane bp = new BorderPane();

        // creating top of pane
        TabPane centerTP = new TabPane();

        Tab sharedTab = setSharedTab();
        Tab localTab = setLocalTab();

        centerTP.getTabs().addAll(sharedTab, localTab);

        Button downloadButton = setDownloadButton(sharedTab, localTab);
        Button uploadButton = setUploadButton(sharedTab, localTab);

        GridPane topGP = new GridPane();
        topGP.add(downloadButton, 0, 0);
        topGP.add(uploadButton, 1, 0);

        bp.setTop(topGP);
        bp.setCenter(centerTP);

        return bp;

    }

    public Tab setSharedTab() throws IOException {

        FileServerClient clientInstance1 = new FileServerClient("DIR");

        Tab sharedTab = new Tab();
        sharedTab.setText("Shared Directory");
        sharedTab.setClosable(false);

        GridPane sharedGP = new GridPane();

        // creating list of files

        String[] sharedFileNames = clientInstance1.getFileNames();
        String[] sharedFileSizes = clientInstance1.getFileSizes();
        String[] sharedLastUpdates = clientInstance1.getLastUpdates();
        String[] sharedFileAuthors = clientInstance1.getFileAuthors();

        ListView<String> sharedFileView = new ListView<>();

        String[] sharedFileViewElements = new String[sharedFileNames.length];

        for (int x = 0; x < sharedFileViewElements.length; x++) {
            sharedFileViewElements[x] = sharedFileNames[x] + "\t\t\t\t" + sharedFileSizes[x] + "\t\t\t\t" + sharedLastUpdates[x] + "\t\t\t\t" + sharedFileAuthors[x];
        }

        sharedFileView.getItems().addAll(sharedFileViewElements);

        sharedFileView.getSelectionModel().selectedItemProperty().addListener(new ChangeListener<String>() {

            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {

                int selectedIndex = sharedFileView.getSelectionModel().getSelectedIndex();


                selectedFileName = sharedFileNames[selectedIndex];
                selectedFileSize = sharedFileSizes[selectedIndex];
                selectedFileAuthor = sharedFileAuthors[selectedIndex];

            }
        });

        sharedFileView.setPrefWidth(1280);
        sharedFileView.setPrefHeight(700);

        sharedGP.add(sharedFileView, 0, 0);
        sharedTab.setContent(sharedGP);

        return sharedTab;

    }

    public Tab setLocalTab() throws IOException {

        LocalDirectory lc = new LocalDirectory();

        Tab localTab = new Tab();
        localTab.setText("Local Directory");
        localTab.setClosable(false);

        GridPane localGP = new GridPane();

        // creating list of files

        String[] localFileNames = lc.getFileNames();
        String[] localFileSizes = lc.getFileSizes();
        String[] localLastUpdates = lc.getLastUpdated();
        String[] localFileAuthors = lc.getFileAuthor();

        this.localFileNames = localFileNames;
        this.localFileSizes = localFileSizes;
        this.localLastUpdates = localLastUpdates;
        this.localFileAuthors = localFileAuthors;

        ListView<String> localFileView = new ListView<>();

        String[] localFileViewElements = new String[localFileNames.length];

        for (int x = 0; x < localFileViewElements.length; x++) {
            localFileViewElements[x] = localFileNames[x] + "\t\t\t\t" + localFileSizes[x] + "\t\t\t\t" + localLastUpdates[x] + "\t\t\t\t" + localFileAuthors[x];
        }

        localFileView.getItems().addAll(localFileViewElements);

        //localFileView.getSelectionModel().getSelectedIndex();
        localFileView.getSelectionModel().selectedItemProperty().addListener(new ChangeListener<String>() {

            @Override
            public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {

                int selectedIndex = localFileView.getSelectionModel().getSelectedIndex();

                GUI.selectedFileName = localFileNames[selectedIndex];
                GUI.selectedFileSize = localFileSizes[selectedIndex];
                GUI.selectedFileAuthor = localFileAuthors[selectedIndex];

            }
        });

        // ---------------------------------------

        localFileView.setPrefWidth(1280);
        localFileView.setPrefHeight(700);

        localGP.add(localFileView, 0, 0);
        localTab.setContent(localGP);

        return localTab;

    }

    public Button setDownloadButton(Tab sharedTab, Tab localTab) throws IOException {

        Button downloadButton = new Button("DOWNLOAD");

        downloadButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {

                // can't download a file that is already local
                if (localTab.isSelected()) {
                    System.out.println("This file is already in the local directory");
                }

                // get the file selected and download it from the server
                else if (sharedTab.isSelected()){
                    try {

                        // send download signal to server
                        String selectedFileName = GUI.selectedFileName;
                        String selectedFileAuthor = GUI.selectedFileAuthor;
                        command = "DOWNLOAD " + selectedFileName;
                        FileServerClient clientDownloadInstance = new FileServerClient(command);

                        // receive file from server

                    } catch (IOException e) {
                        System.err.println("FAILED IN CREATING DOWNLOAD INSTANCE OF CLIENT");
                    }
                } else {
                    System.err.println("FATAL SYSTEM ERROR, CANNOT DOWNLOAD FILE");
                }

            }
        });

        return downloadButton;

    }

    public Button setUploadButton(Tab sharedTab, Tab localTab) throws IOException {

        Button uploadButton = new Button("UPLOAD");
        uploadButton.setOnAction(new EventHandler<ActionEvent>() {
            @Override
            public void handle(ActionEvent event) {

                // can't upload a file that is already shared
                if (sharedTab.isSelected()) { }

                // get the file selected and upload it to the server
                else if (localTab.isSelected()){

                    try {

                        // send download signal to server
                        String selectedFileName = GUI.selectedFileName;
                        String selectedFileAuthor = GUI.selectedFileAuthor;
                        command = "UPLOAD " + selectedFileName;
                        FileServerClient clientUploadInstance = new FileServerClient(command);
                    } catch (IOException e) {
                        System.err.println("FAILED IN CREATING UPLOAD INSTANCE OF CLIENT");
                    }

                } else {

                }

            }
        });

        return uploadButton;

    }

    public static void main(String[] args) { launch(args); }

}
