#coding=utf-8
import pymysql
from optOnMysql import *
from DocumentUnit import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class DocumentsOnMysql(object):
    def __init__(self):
        self.opt_OnMySql = OptOnMysql()

    def exeQuery(self, sql):
        cur = self.opt_OnMySql.exeQuery(sql)
        it = cur.fetchall()
        if it == None:
            # print("there is nothing found")
            return None
        else:
            # print(it[5])
            # print('\n       '.join(it[5].split('|')))
            return it

    def findById(self,id):
        cur = self.opt_OnMySql.exeQuery("select * from document where _id = '%d'" %id)
        it = cur.fetchone()
        # print(it)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")#
        if it == None:
            # print("there is nothing found")
            return 0
        else:
            # print(it[5])
            print('\n       '.join(it[5].split('|')))
            return 1

    def getById(self, id):
        cur = self.opt_OnMySql.exeQuery("select * from document where _id = '%d'" % id)
        it = cur.fetchone()
        # print(it)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")#
        if it == None:
            # print("there is nothing found")
            return None
        else:
            # print(it[5])
            # print('\n       '.join(it[5].split('|')))
            return it

    def findbycriminal(self, crim):
        '''
        :param crim: 罪名
        :return: criminal cursor of data
        '''
        cur = self.opt_OnMySql.exeQuery("select * from document where criminal = '%s'" %crim)
        it = cur.fetchall()
        if it == None:
            print("No data for %s" %crim)
        else:
            print("return data")
            return it







    def findall(self):
        cur = self.opt_OnMySql.exeQuery("select * from document")
        it = cur.fetchall()
        if it == None:
            print("this is now data")
            return it
        else:
            print("fetch all data")
            return it

    def insertOneDocuments(self,document_unit):
        '''
            description:
                 insert one document into mysql
            input:
                document_unit:
                    dict of document to be inserted
            output:
                num of insert document
        '''
        # self.old_document = self.findById(document_unit['_id'])
        # if self.old_document:
        #
        #     print("document exists")
        #     return 1
        # else:
        #     # print("++++++++~~~~~~~~=================----------------")
        sta = 0
        try:
            sql = "insert into document (title, court, url, content, criminal, date) values('{0}', '{1}', '{2}', '{3}', '{4}', '{5}')".\
                format(document_unit["title"],
                       document_unit["court"],
                       document_unit["url"],
                       document_unit["content"].encode('utf8'),
                       document_unit["criminal"],
                       document_unit["date"])
            sta = self.opt_OnMySql.exeUpdate(sql)
            # print(sql)
            # if sta == 1:
            # print("insert success!")
        except:
                print("insert failed!")
                print("the content length is :"+ str(len(document_unit["content"])))
                for i,k in document_unit.items():
                    print(str(i) + " : " + str(k))
                    print(len(k))
                    # print(k)
        # sta = self.opt_OnMySql.exeUpdate("insert into test (id,title) values(%d"%(int(document_unit["id"]))+",'"+document_unit["title"]+"')")
        return sta

    def deleteById(self,id):#
        sta = self.opt_OnMySql.exeDeleteById("delete from document where id='%d'"%id)
        return sta

    def deleteByIds(self,ids):
        sta = 0
        for eachID in ids:
            sta += self.deleteById(eachID)
        return sta

    def connClose(self):
        self.opt_OnMySql.connClose()

if __name__ == "__main__":
    # print(1)
    document_unit = dict()
    # document_unit["_id"] = 2
    document_unit["title"] = "叶某交通肇事罪二审刑事裁定书"
    document_unit["court"] = "安徽省合肥市中级人民法院"
    document_unit["date"] = "2015-06-05"
    document_unit["url"] = "http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx?DocID=eff7f53c-b647-11e3-84e9-5cf3fc0c2c18"
    # print(len(document_unit["url"]))
    document_unit["content"] = "安徽省合肥市中级人民法院|刑 事 裁 定 书|（2015）合刑终字第00256号|原公诉机关合肥市瑶海区人民检察院。|上诉人（原审被告人）叶某，无业。因涉嫌犯交通肇事罪于2014年12月9日被合肥市公安局取保候审，2015年3月27日被合肥市瑶海区人民法院取保候审，同年4月10日被合肥市瑶海区人民法院决定逮捕，同日由合肥市公安局执行逮捕。现羁押于合肥市第一看守所。|合肥市瑶海区人民法院审理合肥市瑶海区人民检察院指控原审被告人叶某犯交通肇事罪一案，于2015年4月13日作出（2015）瑶刑初字第00293号刑事判决。原审被告人叶某不服，提出上诉。本院依法组成合议庭，经阅卷、讯问上诉人，认为本案事实清楚，决定不开庭审理。本案现已审理终结。|原判认定：2014年11月13日2时50分许，被告人叶某醉酒后驾驶赣E×××××号宝马轿车，沿合肥市北一环路由西向东行驶至瑶海区站西路桥下穿桥附近时，因操作不当，赣E×××××号轿车碰撞到道路中间隔离护栏，导致方向失控后又碰撞到道路南侧的防护墙，造成被告人叶某及车辆乘坐人徐某、潘某受伤及车辆受损、隔离栏损坏。被害人徐某（女，1992年2月15日出生）经医院抢救无效于2014年12月3日死亡。经公安机关认定，被告人叶某承担此次事故的全部责任。|另查明，事故发生后，合肥市公安局交通警察支队瑶海大队民警接到附近群众电话报警后赶至案发现场，被告人叶某及被害人徐某已被120救护车送往医院救治，公安民警对现场勘查完毕后，遂赶至合肥市第二人民医院对被告人叶某抽血待检，后将其带至公安机关接受调查。经检测，被告人叶某血液中乙醇含量为118mg／100ml，属醉酒驾驶。经安徽同德司法鉴定所鉴定，被害人徐某系道路交通事故致特重度颅脑损伤继发多器官功能衰竭死亡。案发后，被告人叶某与被害人徐某的亲属达成赔偿协议，除已支付的医疗费外，由被告人叶某另外一次性赔偿给被害人徐某亲属各项经济损失共计人民币42万元（已即时付清），被害人徐某的亲属对被告人叶某表示谅解，建议司法机关对其免予刑事处罚。|原判认定上述事实的证据有：被告人叶某的供述、证人刘某、潘某、李某的证言、辨认笔录、现场勘验检查笔录、现场图及照片、道路交通事故认定书、尸体检验报告、人体乙醇含量检验报告书、涉案车辆交通事故技术鉴定意见书、死亡医学证明书、医院门诊病历、视听资料、机动车驾驶证及行驶证查询记录、情况说明、归案经过、户籍证明、协议书、谅解书、银行汇款单据等。|原判认为：被告人叶某违反交通运输管理法规，醉酒后驾驶机动车辆在道路上行驶，且操作不当，以致发生交通事故，造成一人死亡的严重后果，并负事故的全部责任，其行为已构成交通肇事罪。被告人叶某归案后如实供述自己的罪行，庭审中自愿认罪，可从轻处罚。被告人叶某案发后积极赔偿被害人徐某亲属的经济损失，并取得被害人亲属的谅解，可酌情从轻处罚。但被告人叶某醉酒后驾驶机动车发生交通事故，又可酌情从重处罚。依照《中华人民共和国刑法》第一百三十三条、第六十七条第三款、第六十一条规定，判决：被告人叶某犯交通肇事罪，判处有期徒刑十个月。|原审被告人叶某的上诉请求和理由为：一审量刑过重。|经审理查明：原判认定上诉人叶某犯交通肇事罪的事实，已被一审判决列举的证据证实，所列证据经一审当庭举证、质证，合法有效。本院审理中，上诉人叶某未提出新的证据，本院对一审判决认定的事实及相关证据予以确认。|关于上诉人叶某认为一审量刑过重的上诉理由，经查，一审在对上诉人叶某量刑时已考虑其如实供述、自愿认罪、取得被害人亲属谅解等情节，综合量刑对其从轻处罚，判决并无不当，上诉人叶某的上诉理由不能成立，本院不予支持。|本院认为：上诉人叶某违反交通运输管理法规，醉酒后驾驶机动车辆在道路上行驶，操作不当发生交通事故，造成一人死亡，负事故的全部责任，其行为已构成交通肇事罪，依法应予惩处。上诉人叶某醉酒后驾驶机动车发生交通事故，负事故的全部责任，可酌情从重处罚。上诉人叶某归案后如实供述自己的罪行，自愿认罪，可从轻处罚。上诉人叶某案发后积极赔偿被害人亲属的经济损失，并取得被害人亲属的谅解，可酌情从轻处罚。原判认定上诉人叶某犯交通肇事罪的事实清楚，证据确实充分，适用法律正确，量刑适当。审判程序合法。依照《中华人民共和国刑事诉讼法》第二百二十五条第一款第（一）项之规定，裁定如下：|驳回上诉，维持原判。|本裁定为终审裁定。|审　判　长　　张　恒|审　判　员　　陆文波|代理审判员　　董雪美|二〇一五年六月五日|书　记　员　　黄圣全|附：相关法律条文|《中华人民共和国刑事诉讼法》第二百二十五条第一款第（一）项原判决认定事实和适用法律正确、量刑适当的，应当裁定驳回上诉或者抗诉，维持原判。"
    document_unit["criminal"] = "交通肇事罪"

    # print(document_unit)
    opt = DocumentsOnMysql()

    # opt.insertOneDocuments(document_unit)

    # it = opt.findbycriminal(u"交通肇事罪")
    # it = opt.findall()
    # print(len(it))
    # j = 1
    # for i in it:
            # print(j)
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # print(''i[5].split('|'))
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            # j += 1
    # opt.connClose()
    # it = opt.findall()
    # print(len(it[0]))
    # opt.findById(2005)
# update document set content = "云南省高级人民法院|刑 事 附 带 民 事 裁 定 书|（2015）云高刑终字第477号|原公诉机关云南省昆明市人民检察院。上诉人（原审被告人）贺顺林，无业。2014年4月7日因本案被刑事拘留，同年4月21日被逮捕。现羁押于昆明市官渡区看守所。辩护人凌艳传、李永交，云南汇同律师事务所律师。原审附带民事诉讼原告人郑某，农民。系被害人杜某戊之妻。原审附带民事诉讼原告人杜某甲。系被害人杜某戊之子。原审附带民事诉讼原告人杜某乙。系被害人杜某戊之子。原审附带民事诉讼原告人杜某丙。系被害人杜某戊之子。原审附带民事诉讼原告人杜某丁。系被害人杜某戊之女。云南省昆明市中级人民法院审理昆明市人民检察院指控原审被告人贺顺林犯故意杀人罪，原审附带民事诉讼原告人郑某、杜某甲、杜某丙、杜某乙、杜某丁提起附带民事诉讼一案，于二O一五年一月四日作出（2014）昆刑一初字第113号刑事附带民事判决。原审被告人贺顺林不服，提出上诉。本院依法组成合议庭，经过阅卷、审查上诉意见，听取辩护意见并依法讯问被告人，认为本案事实清楚，决定不开庭审理。现已审理终结。原判认定，2014年4月5日凌晨，被告人贺顺林将其号牌为云A·168Z1的中华牌轿车停放在昆明市官渡区螺峰村133号杜某戊家门口南侧，杜某戊认为贺顺林停车堵到自家门口，遂让其挪车，双方为此事发生争执。过程中贺顺林用随身携带的刀具刺中杜某戊胸部一刀后逃离现场。被害人杜某戊因心脏破裂经送医院抢救无效死亡。同年4月7日13时许，被告人贺顺林向昆明市公安局投案。原判根据上述事实和在案证据，依照《中华人民共和国刑法》第二百三十二条、第六十七条第一款、第五十七条第一款、第三十六和《中华人民共和国刑事诉讼法》第九十九条及《最高人民法院关于适用〈中华人民共和国刑事诉讼法〉的解释》第一百五十五条第一、二款之规定，认定被告人贺顺林犯故意杀人罪，判处无期徒刑，剥夺政治权利终身；判令被告人贺顺林赔偿附带民事诉讼原告人郑某、杜某甲、杜某丙、杜某乙、杜某丁经济损失人民币27938.5元。宣判后，原审被告人贺顺林上诉称其行为不构成故意杀人罪，应当以过失致人死亡罪对其定罪量刑；有自首情节，归案后认罪态度好，已对被告人家属赔偿了20000元，请求对其从轻处罚；并提出一审附带民事部分判决赔偿过高。其辩护人除以相同理由为其辩护外，还提出受害人对该事件的发生具有重大过错的辩护意见。经审理查明，原判认定2014年4月5日凌晨，被告人贺顺林与被害人杜某戊因挪车一事发生争执。期间，贺顺林用其随身携带的刀具刺中杜某戊胸部一刀后逃离现场。杜某戊因心脏破裂经送医院抢救无效死亡的事实清楚。该事实有下列证据予以证实：1.受案登记表、归案经过，证实本案来源及案发后被告人贺顺林到公安机关自动投案的事实。2.现场勘查笔录及照片、现场辨认笔录及照片，证实案发现场情况。贺顺林归案后对作案现场、丢弃作案工具的现场进行了指认。3.情况说明证实：⑴贺顺林投案后，公安机关依法提取其所穿沾附血迹的毛衣和鞋子；⑵公安机关根据贺顺林的供述，到其丢弃作案工具的地点查找，未查获凶器。4.DNA鉴定书证实：⑴现场“常来”客栈大门南侧地面上提取的血迹检见人血，与被害人杜某戊的基因分型一致；⑵贺顺林作案时所穿毛衣和鞋子上的可疑血迹检见人血，与贺顺林的基因分型一致。5.证人证言⑴杜某丙证实：当时我在家听到父亲杜某戊在家门口和一男子吵架，下楼看见潘姓女邻居扶着全身是血的杜某戊，他说被杀了一刀，叫我们送他去医院。潘姓女邻居说是她丈夫因挪车的问题刺到杜某戊。后杜某戊经抢救无效死亡。⑵潘丽英证实：我和前夫贺顺林居住在官渡区螺峰村117号附1号。案发当晚，我听到贺顺林与邻居杜某戊吵架，我赶下楼的时候杜某戊正捂着肚子，贺顺林的车停在杜某戊家客栈门口。我和同时赶到的杜某丙查看杜某戊的情况，贺顺林从我身旁走过，没和我说话。杜某戊说他让贺顺林挪一下车不要挡在他家客栈门口，结果被贺顺林杀到。⑶甘乐证实：我在“常来”客栈听到房东杜某戊说话的声音，好像是杜某戊叫对方男子不要把车停得太久，男子和杜某戊争执，我出去劝他们，男子准备要走了，我就折回客栈，但他们又吵起来，当时他们两人贴得很近，杜某戊好像被男子推了一下，我又出去劝他们，杜某戊胸口流血，说他心脏被刺到了，叫我打120，那个男子叫我不要打，然后他就走了，与杜某戊争执的男子平时也会把他的车停在“常来”客栈门口。案发当天两人开始争执的时候，只有我在场，男子走后，男子的妻子和杜某戊的儿子就先后过来了。经甘乐对照片混同辨认，确认和杜某戊发生争执的男子即被告人贺顺林。6.尸体检验报告及照片证实被害人杜某戊系心脏破裂死亡。7.被告人贺顺林对案发当晚其因停车一事与杜某戊发生纠纷，后用自己随身携带的折叠刀刺到杜某戊胸部的犯罪事实供认不讳。但辩称不是自己主动刺的，是他拿着刀时杜某戊自己转身撞上来的。8.户籍证明证实被告人贺顺林的基本身份情况。以上证据均经一审庭审质证，证据来源合法，内容客观、真实，本院予以确认。本院认为，上诉人贺顺林仅因琐事争执即持刀故意捅刺他人，并致被害人死亡的严重后果，其行为已构成故意杀人罪，应依法惩处。贺顺林犯罪后自动投案并如实供述自己的罪行，是自首，依法可以从轻或减轻处罚。关于贺顺林及其辩护人所提应以过失致人死亡罪对其定罪量刑的上诉理由及辩护意见，经查，贺顺林仅因口角便用利刃直接刺向被害人胸部，并致被害人心脏破裂死亡，从其使用的工具、刺杀的部位、力度、造成的后果来看，其行为应当构成故意杀人罪，故对此意见不予采纳；所提有自首情节、已赔偿被害人亲属部分经济损失的理由经查属实，但原判已鉴于此情节对其从轻处罚，现再以相同事由要求从轻处罚的意见不予采纳。关于贺顺林的辩护人所提被害人对该事件的发生具有重大过错的意见，与在案证据证明事实不符，不能成立，本院不予采纳。关于贺顺林所提原判附带民事部分判决赔偿金额过高的理由，经查，原判根据相关法律规定，结合被害人所遭受的物质损失，所作判决赔偿金额并无不当，对该意见不予支持。综上，原判定罪准确，量刑及民事判赔适当。审判程序合法。据此，依照《中华人民共和国刑事诉讼法》第二百二十五条第一款（一）项和《中华人民共和国民事诉讼法》第一百七十条第一款（一）项的规定，裁定如下：驳回上诉，维持原判。本裁定为终审裁定。审判长杨晓娅审判员邹浪萍代理审判员张赵琳二〇一五年五月八日书记员李静
# " where _id = 2336;