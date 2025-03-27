package Client;

import javafx.scene.control.ListCell;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class LocalFileListView {

    public static class LocalFile {

        private static String fileName;
        private static String fileSize;
        private static String lastUpdated;
        private static String fileAuthor;

        public String getFileName() { return fileName; }
        public String getFileSize() { return fileSize; }
        public String getLastUpdated() { return lastUpdated; }
        public String getFileAuthor() { return fileAuthor; }

        public LocalFile(String name, String size, String updated, String author) {
            super();
            this.fileName = name;
            this.fileSize = size;
            this.lastUpdated = updated;
            this.fileAuthor = author;
        }

    }

    public static class LocalFileListCell extends ListCell<LocalFileListView.LocalFile> {

        public static GridPane fileListCellPane;
        public static Text fileNameText;
        public static Text fileSizeText;
        public static Text lastUpdatedText;
        public static Text fileAuthorText;

        // Constructor
        public LocalFileListCell() {
            super();
            fileListCellPane = new GridPane();


            fileNameText = new Text(LocalFileListView.LocalFile.fileName);
            fileSizeText = new Text(LocalFileListView.LocalFile.fileSize);
            lastUpdatedText = new Text(LocalFileListView.LocalFile.lastUpdated);
            fileAuthorText = new Text (LocalFileListView.LocalFile.fileAuthor);

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
        public void updateItem(LocalFileListView.LocalFile file, boolean empty) {
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
