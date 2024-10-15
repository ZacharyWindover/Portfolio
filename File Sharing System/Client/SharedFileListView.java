package Client;

import javafx.scene.control.ListCell;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class SharedFileListView {

    public static class SharedFile {

        private static String fileName;
        private static String fileSize;
        private static String lastUpdated;
        private static String fileAuthor;

        public static String getFileName() { return fileName; }
        public static String getFileSize() { return fileSize; }
        public String getLastUpdated() { return lastUpdated; }
        public static String getFileAuthor() { return fileAuthor; }

        public SharedFile(String name, String size, String updated, String author) {
            super();
            this.fileName = name;
            this.fileSize = size;
            this.lastUpdated = updated;
            this.fileAuthor = author;
        }

    }

    public static class SharedFileListCell extends ListCell<SharedFileListView.SharedFile> {

        public static GridPane fileListCellPane;
        public static Text fileNameText;
        public static Text fileSizeText;
        public static Text lastUpdatedText;
        public static Text fileAuthorText;

        // Constructor
        public SharedFileListCell() {
            super();
            fileListCellPane = new GridPane();


            fileNameText = new Text(SharedFileListView.SharedFile.fileName);
            fileSizeText = new Text(SharedFileListView.SharedFile.fileSize);
            lastUpdatedText = new Text(SharedFileListView.SharedFile.lastUpdated);
            fileAuthorText = new Text (SharedFileListView.SharedFile.fileAuthor);

            /*
            fileNameText = new Text();
            fileSizeText = new Text();
            lastUpdatedText = new Text();
            fileAuthorText = new Text();

             */

            fileListCellPane.add(fileNameText, 0, 0);
            fileListCellPane.add(fileSizeText, 1, 0);
            fileListCellPane.add(lastUpdatedText, 2, 0);
            fileListCellPane.add(fileAuthorText, 3, 0);

            fileListCellPane.setHgap(50);

        }


        // Updating cell
        public void updateItem(SharedFileListView.SharedFile file, boolean empty) {
            super.updateItem(file, empty);

            if (file != null && !empty) {

                fileNameText.setText(file.getFileName());
                fileSizeText.setText(file.getFileSize());
                lastUpdatedText.setText(file.getLastUpdated());
                fileAuthorText.setText(file.getFileAuthor());


                //fileListCellPane = new GridPane();

                //fileListCellPane.add(fileNameText, 0, 0);
                //fileListCellPane.add(fileSizeText, 1, 0);
                //fileListCellPane.add(lastUpdatedText, 2, 0);
                //fileListCellPane.add(fileAuthorText, 3, 0);

                //fileListCellPane.setHgap(50);


                setGraphic(fileListCellPane);

            }

            else {
                setGraphic(null);
            }
        }

    }

}
