from segmentation_new.cnn import FCN32
from segmentation_new.cars_loader import CarsLoader


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = CarsLoader.load_data()

    net = FCN32()
    net.fit(train_x, train_y, test_x, test_y, nb_epochs=30)
    net.save()
