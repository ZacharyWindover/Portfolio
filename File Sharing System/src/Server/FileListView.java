package Server;

import javafx.scene.control.ListCell;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

public class FileListView {

    public static class File {

        private static String fileName;
        private static String fileSize;
        private static String lastUpdated;
        private static String fileAuthor;

        public String getFileName() { return fileName; }
        public String getFileSize() { return fileSize; }
        public String getLastUpdated() { return lastUpdated; }
        public String getFileAuthor() { return fileAuthor; }

        public File(String name, String size, String updated, String author) {
            super();
            this.fileName = name;
            this.fileSize = size;
            this.lastUpdated = updated;
            this.fileAuthor = author;
        }

    }

    public static class FileListCell extends ListCell<FileListView.File> {

        public static GridPane fileListCellPane;
        public static Text fileNameText;
        public static Text fileSizeText;
        public static Text lastUpdatedText;
        public static Text fileAuthorText;

        // Constructor
        public FileListCell() {
            super();
            fileListCellPane = new GridPane();

            fileNameText = new Text(File.fileName);
            fileSizeText = new Text(File.fileSize);
            lastUpdatedText = new Text(File.lastUpdated);
            fileAuthorText = new Text (File.fileAuthor);

            fileListCellPane.add(fileNameText, 0, 0);
            fileListCellPane.add(fileSizeText, 1, 0);
            fileListCellPane.add(lastUpdatedText, 2, 0);
            fileListCellPane.add(fileAuthorText, 3, 0);

            fileListCellPane.setHgap(50);

        }

        // Updating cell
        protected void updateItem(FileListView.File file, boolean empty) {
            super.updateItem(file, empty);

            if (file != null && !empty) {
                fileNameText.setText(file.getFileName());
                fileSizeText.setText(file.getFileSize());
                lastUpdatedText.setText(file.getLastUpdated());
                fileAuthorText.setText(file.getFileAuthor());

                setGraphicTextGap(25);
                setGraphic(fileListCellPane);

            }

            else {
                setGraphic(null);
            }
        }

    }
}
