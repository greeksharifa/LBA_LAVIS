import argparse

from torch.utils.data import DataLoader

from dataset.VQA_Introspect import VQAIntrospectDataset

def get_args():
    parser = argparse.ArgumentParser(description='LBA method')
    # path
    parser.add_argument('--dataset_path', type=str, default='/data1/VQA-Introspect/VQAIntrospect_valv1.0.json', help='path to the dataset')
    # vis_root
    parser.add_argument('--vis_root', type=str, default='/data1/coco/images/', help='Root directory of images (e.g. coco/images/)')
    
    '''
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')
    '''

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    
    

    # Create an instance of your dataset
    dataset = VQAIntrospectDataset(None, None, args.vis_root, [args.dataset_path, '/data1/VQA/v2/v2_mscoco_val2014_annotations.json'])

    # Create a DataLoader to load the dataset
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collater)
    
    for i, batch in enumerate(dataloader):
        print('i:', i)
        from pprint import pprint
        pprint(batch, width=300)
        
        
        import requests
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")#, device_map="auto")

        images = batch['image']
        questions = batch['text_input']
        inputs = processor(images, questions, return_tensors="pt", padding=True).to("cuda")#, torch.float16)

        out = model.generate(**inputs)
        print('-' * 20 + 'result' + '-' * 20)
        print(out)
        print(processor.batch_decode(out, skip_special_tokens=True))
        
        
        break
    
    
    
if __name__ == '__main__':
    main()
    