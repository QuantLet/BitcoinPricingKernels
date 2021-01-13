import sshtunnel
import pymongo
import pprint
import datetime
import pandas as pd
import os

# For Deribit
import asyncio
import websockets
import json

# Avg IV per Day
from bson.json_util import dumps # dump Output

# Todo: connection doesnt stop atm
class BRC:
    def __init__(self):
        
        self.MONGO_HOST = '35.205.115.90'
        self.MONGO_DB   = 'cryptocurrency'
        self.MONGO_USER = 'winjules2'
        self.MONGO_PASS = ''
        self.PORT = 27017
        #self.server_started = self._start()
        #if self.server_started:
        print('\nGoing Local')
        self.client = pymongo.MongoClient('localhost', 27017) 
        self.db = self.client[self.MONGO_DB]
        self.collection = self.db['deribit_orderbooks']
        self._generate_stats()
        #self._mean_iv(do_sample = False, write_to_file = False)

    def _server(self):
        self.server = sshtunnel.SSHTunnelForwarder(
            self.MONGO_HOST,
            ssh_username=self.MONGO_USER,
            ssh_password=self.MONGO_PASS,
            remote_bind_address=('127.0.0.1', self.PORT)
            )
        return self.server

    def _start(self):
        #self.server = self._server()
        #self.server.start()
        return True

    def _stop(self):
        #self.server.stop()
        return True

    def _filter_by_timestamp(self, starttime, endtime):        
        """
        Example:
        starttime = datetime.datetime(2020, 4, 19, 0, 0, 0)
        endtime = datetime.datetime(2020, 4, 20, 0, 0, 0)
        """
        ts_high     = round(endtime.timestamp() * 1000)
        ts_low      = round(starttime.timestamp() * 1000)
        return ts_high, ts_low

    def _generate_stats(self):
        print('\n Established Server Connection')
        print('\n Available Collections: ', self.db.collection_names())
        print('\n Size in GB: ', self.db.command('dbstats')['dataSize'] * 0.000000001) 
        print('\n Object Count: ', self.collection.count())
        
        # Get first and last element:
        last_ele = self.collection.find_one(
        sort=[( '_id', pymongo.DESCENDING )]
        )

        first_ele = self.collection.find_one(
            sort = [('_id', pymongo.ASCENDING)]
        )
        
        self.first_day = datetime.datetime.fromtimestamp(round(first_ele['timestamp']/1000))
        self.last_day  = datetime.datetime.fromtimestamp(round(last_ele['timestamp']/1000))
        self.first_day_timestamp = first_ele['timestamp']
        self.last_day_timestamp  = last_ele['timestamp']

        print('\n First day: ', self.first_day, ' \n Last day: ', self.last_day)

    def synth_btc(self, do_sample, write_to_file):
        """
        Extract high frequency prices of Deribit synthetic btc price
        """
        print('extracting synth index')
        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%M", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        else:
            pipeline = [
                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d-%H-%m", "date": {'$toDate': '$timestamp' } }},
                            'avg_btc_price': {'$avg': '$underlying_price'}
                        }
                }

            ]
        
        print('Pumping the Pipeline')
        synth_per_minute = self.collection.aggregate(pipeline)

        # Save Output as JSON
        a = list(synth_per_minute)
        j = dumps(a, indent = 2)
        
        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "/Users/julian/src/spd/out/synth_btc_per_minute.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out



    def _mean_iv(self, do_sample = False, write_to_file = False):
        """
        Task: 
            Select Average IV (for bid and ask) and group by day!

        Paste in the pipeline to have a sample for debugging
            {
                "$sample": {"size": 10},
            },
        """
        print('init mean iv')

        if do_sample:
            pipeline = [
                {
                    "$sample": {"size": 120000},
                },

                # Try to Subset / WHERE Statement
                {'$match': {'bid_iv': {"$gt": 0.02}}},

                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid': {'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]
        else:
            pipeline = [
                {
                    '$group':{"_id": { "$dateToString": { "format": "%Y-%m-%d", "date": {'$toDate': '$timestamp' } }},
                            'avg_ask': {'$avg': '$ask_iv'},
                            'avg_bid':{'$avg': '$bid_iv'},
                            'avg_btc_price': {'$avg': '$underlying_price'}

                        }
                }

            ]
        
        print('Pumping the Pipeline')
        avg_iv_per_day = self.collection.aggregate(pipeline)
        #print(list(avg_iv_per_day))

        # Save Output as JSON
        a = list(avg_iv_per_day)
        j = dumps(a, indent = 2)

        if write_to_file:
            # Dump each element in a dict and save as JSON
            fname = "/Users/julian/src/spd/out/volas_per_day.JSON"
            print('Writing Output to ', fname)
            out = {}
            for ele in a:
                out[ele['_id']] = {'ask': ele['avg_ask'],
                                    'bid': ele['avg_bid'],
                                    'underlying': ele['avg_btc_price']}

            with open(fname,"w") as f:
                json.dump(out, f)
        else:
            out = json.loads(j)

        return out

    # Deribit
    def create_msg(self, _tshigh, _tslow):
        # retrieves constant interest rate for time frame
        self.msg = \
        {
        "jsonrpc" : "2.0",
        "id" : None,
        "method" : "public/get_funding_rate_value",
        "params" : {
            "instrument_name" : "BTC-PERPETUAL",
            "start_timestamp" : _tslow,
            "end_timestamp" : _tshigh
            }
        }
        return None

    async def call_api(self):
        async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
            await websocket.send(json.dumps(self.msg))
            while websocket.open:
                print(self.msg)
                response = await websocket.recv()
                # do something with the response...
                self.response = json.loads(response)
                self.historical_interest_rate = round(self.response['result'], 4)
                return None

    def _run(self, starttime, endtime, download_interest_rates, download_historical_iv):
        
        #server_started = self._start()
        
    
        try:

            download_starttime = datetime.datetime.now()

            ts_high, ts_low = self._filter_by_timestamp(starttime, endtime)
            res = self.collection.find({ "$and": [{'timestamp': {"$lt": ts_high}},
                                {'timestamp': {"$gte": ts_low}}]})#.sort('timestamp')
            
            nresults = res.count()
            if nresults == 0:
                raise ValueError('No DB results returned, proceeding with the next day.')
            else:
                print('DB count for current day', nresults)

            out = []
            for doc in res:
                out.append(doc)
                #print(doc)

            if download_interest_rates:
                try:
                    self.create_msg(ts_high, ts_low)
                    asyncio.get_event_loop().run_until_complete(self.call_api())
                except Exception as e:
                    print('Error while downloading from Deribit: ', e)
                    print('Proceeding with interest rate of 0')
                    self.historical_interest_rate = 0

            download_endtime = datetime.datetime.now()
            # Got to change this to subtr.
            print('\nDownload Time: ', download_endtime - download_starttime)
            print('\nDisconnecting Server')

            return out, self.historical_interest_rate

        except Exception as e:
            print('Error: ', e)
            print('\nDisconnecting Server within error handler')
            self._stop()
            self.client.close()


if __name__ == '__main__':
    brc = BRC()
    dat = brc._run(starttime = datetime.datetime(2020, 4, 14, 0, 0, 0),
                    endtime = datetime.datetime(2020, 4, 15, 0, 0, 0))

    #d = pd.DataFrame(dat)
    #d.to_csv('data/orderbooks_test.csv')
