import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import BertForTokenClassification

from dataset import prepare_data_for_training

bert_model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3)

train_loader, test_loader = prepare_data_for_training()

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'
print(device)

learning_rate = 1e-4
num_epochs = 300
model_saving_path = './bert_keywords.pt'


def train_model(model, dataloaders, optimizer, num_epochs, model_saving_path):
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    since = time.time()
    tb = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        loss_dict = {}
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for _, data in enumerate(dataloaders_dict[phase], 0):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    sent = data['sent_ids'].to(device, dtype=torch.long)
                    y = data['target'].to(device, dtype=torch.long)

                    outputs = model(input_ids=sent.to(device), labels=y.to(device))
                    loss = outputs.loss
                    running_loss += loss.item() * sent.shape[0]

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(epoch_loss)
            loss_dict[phase] = epoch_loss
            model.save_pretrained("bert_keywords")
            torch.save(model.state_dict(), model_saving_path)

        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    torch.cuda.empty_cache()
    bert_model.to(device)
    optimizer = torch.optim.Adam(bert_model.parameters(), lr=learning_rate)
    dataloaders_dict = {'train': train_loader, 'valid': test_loader}
    model_ft = train_model(bert_model, dataloaders_dict, optimizer, num_epochs, model_saving_path)