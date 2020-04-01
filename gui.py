import os
import sys
import json
from PyQt5 import QtCore,QtWidgets,uic
import utils


QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
config_path = 'gui/config.json'
config = json.loads(open(config_path,'rb').read())
Ui_MainWindow, QtBaseClass = uic.loadUiType(config['ui'])

class MainUI(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self._bind_ui_()
        self._init_ui_()


    def _bind_ui_(self):
        self.selectBtn.clicked.connect(self.openfile)
        self.startSearch.clicked.connect(self.start)
        self.resultTable.doubleClicked.connect(self.double_click_cell)
        self.addSearchDir.clicked.connect(self.add_search_dir)
        self.updateIndex.clicked.connect(self.sync_index)
        self.removeInvalidIndex.clicked.connect(self.remove_invalid_index)


    def _init_ui_(self):
        if os.path.exists(utils.name_index_path):
            self.exists_index = utils.get_exists_index()                                                # 加载索引
        self.resultTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)         # 填充显示表格
        self.resultTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)                    # 表格设置只读
        self.searchDirTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.searchDirTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.update_dir_table()


    def openfile(self):
        self.input_path = QtWidgets.QFileDialog.getOpenFileName(self,'选择图片','','Image files(*.*)')
        self.filePath.setText(self.input_path[0])


    def double_click_cell(self, info):
        row = info.row()
        column = info.column()
        os.system(f'"{self.resultTable.item(row, column).text()}"')


    def start(self):
        if (config['search_dir'] == []) or (not os.path.exists(utils.name_index_path)):
            QtWidgets.QMessageBox.information(self, '提示', '索引都没有建搜你🐎 搜')
            return
        self.resultTable.setRowCount(0)                                                                 # 清空表格
        nc = self.resultCount.value()
        results = utils.checkout(self.input_path[0], self.exists_index, nc)
        results = [i for i in results]
        for i in results:
            row = self.resultTable.rowCount()
            self.resultTable.insertRow(row)
            item = QtWidgets.QTableWidgetItem(i)
            self.resultTable.setItem(row,0,item)


    def update_dir_table(self):
        self.searchDirTable.setRowCount(0)
        for i in config['search_dir']:
            row = self.searchDirTable.rowCount()
            self.searchDirTable.insertRow(row)
            item = QtWidgets.QTableWidgetItem(i)
            self.searchDirTable.setItem(row,0,item)


    def add_search_dir(self):
        self.input_path = QtWidgets.QFileDialog.getExistingDirectory(self,'选择一个需要索引的图片目录')
        if not self.input_path:
            return
        config['search_dir'].append(self.input_path)
        self.save_settings()
        self.update_dir_table()


    def remove_invalid_index(self):
        utils.remove_nonexists()
        self.exists_index = utils.get_exists_index()
        QtWidgets.QMessageBox.information(self, '提示', '无效索引已删除')


    def sync_index(self):
        utils.remove_nonexists()
        for image_dir in config['search_dir']:
            exists_index = utils.index_target_dir(image_dir)
            utils.update_ir_index(exists_index)
        self.exists_index = utils.get_exists_index()
        QtWidgets.QMessageBox.information(self, '提示', '索引同步已完成')


    def save_settings(self):
        with open(config_path, 'wb') as wp:
            wp.write(json.dumps(config, indent=2, ensure_ascii=False).encode('UTF-8'))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainUI()
    window.show()
    sys.exit(app.exec_())
